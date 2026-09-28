"""Render-time estimate on a remote GPU worker (docs/adr/render-time-estimate.md).

The control plane times every remote synthesis task end to end (dispatch to
result read back) and files it under the worker's own bucket; the estimate
reads only the bucket of the target the render will actually go to; a chosen
worker that cannot take work gets a note, never a number; a remote longform
render counts down per chapter.
"""
import asyncio
import importlib
import json
import sqlite3
import time
import types

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from worker.lifecycle import Attempt, AttemptState, Task, TaskState
from worker.routing import Decision, Target

WORKER = "cbd2b78f7ddc"
GPU = "NVIDIA GeForce RTX 5080"
DEVICE = f"remote:{WORKER}:rtx-5080"
REMOTE = Decision(remote=True, worker_id=WORKER, label="KAIROS", reason="chosen")


@pytest.fixture(autouse=True)
def mods():
    """The live modules (tests/backend/conftest.py purges core.* / services.*
    between its tests, so collection-time aliases can be stale)."""
    return types.SimpleNamespace(
        core_db=importlib.import_module("core.db"),
        device_caps=importlib.import_module("core.device_caps"),
        gateway=importlib.import_module("services.gpu_gateway"),
        remote=importlib.import_module("services.render_remote"),
        timing=importlib.import_module("services.render_timing"),
        routing=importlib.import_module("worker.routing"),
    )


@pytest.fixture
def db(tmp_path, monkeypatch, mods):
    path = tmp_path / "remote.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(mods.core_db._BASE_SCHEMA)

    def connect():
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(mods.core_db, "get_db", connect)
    return path


def _rows(path):
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM render_timings ORDER BY id")]


def _target(**kw):
    base = dict(id=WORKER, label="KAIROS", connected=True, available=True, status="ready",
                gpu_name=GPU)
    base.update(kw)
    return Target(**base)


@pytest.fixture
def enrolled(monkeypatch, mods):
    """KAIROS is enrolled and chosen in the GPU picker."""
    state = {"target": _target()}
    monkeypatch.setattr(mods.routing, "get_target_id", lambda: WORKER)
    monkeypatch.setattr(mods.routing, "list_targets",
                        lambda control_plane=None: [mods.routing.local_target(), state["target"]])

    class _Plane:
        running = True
        pool = None
        scheduler = None

    monkeypatch.setattr("worker.service.control_plane", _Plane())
    return state


def _wav(path, seconds, rate=8000):
    sf.write(path, np.zeros(int(seconds * rate), dtype=np.float32), rate)
    return str(path)


# ── Identity ──────────────────────────────────────────────────────────────────


def test_the_bucket_names_the_worker_and_its_gpu(mods):
    assert mods.remote.gpu_slug(GPU) == "rtx-5080"
    assert mods.remote.gpu_slug("NVIDIA GeForce RTX 5080 Laptop GPU") == "rtx-5080-laptop"
    assert mods.remote.remote_device(WORKER, GPU) == DEVICE
    assert mods.remote.remote_device(WORKER, "") == f"remote:{WORKER}"
    assert mods.remote.remote_device("", GPU) == ""
    # The timing table keeps the whole remote id (local devices are short).
    assert mods.timing.clean_id(DEVICE, limit=64) == DEVICE


# ── Which bucket the estimate reads ──────────────────────────────────────────


def test_the_route_follows_the_gpu_picker(enrolled, mods):
    route = mods.remote.route_for("tts")
    assert route.kind == "remote" and route.device == DEVICE and route.label == "KAIROS"
    assert route.to_dict() == {"kind": "remote", "label": "KAIROS"}  # never the worker id

    enrolled["target"] = _target(connected=False, available=False, status="offline",
                                 detail="offline")
    offline = mods.remote.route_for("tts")
    assert offline.kind == "unavailable" and offline.offline
    assert offline.to_dict() == {"kind": "unavailable", "label": "KAIROS", "offline": True}

    enrolled["target"] = _target(available=False, status="busy", detail="paused after repeated failures")
    paused = mods.remote.route_for("tts")
    assert paused.kind == "unavailable" and not paused.offline


def test_local_choice_unported_work_and_a_removed_worker_run_here(enrolled, monkeypatch, mods):
    assert mods.remote.route_for("dictation").kind == "local"  # never remote
    monkeypatch.setattr(mods.routing, "list_targets",
                        lambda control_plane=None: [mods.routing.local_target()])
    assert mods.remote.route_for("tts").kind == "local"  # routing falls back here
    monkeypatch.setattr(mods.routing, "get_target_id", lambda: mods.routing.LOCAL)
    assert mods.remote.route_for("tts").kind == "local"


def test_a_routing_failure_is_local(monkeypatch, mods):
    def boom(*a, **k):
        raise RuntimeError("registry locked")

    monkeypatch.setattr(mods.routing, "status", boom)
    assert mods.remote.route_for("tts") is mods.remote.LOCAL_ROUTE


# ── Recording: one row per remote task, end to end ───────────────────────────


class FakeScheduler:
    def __init__(self, *, result_ref, delay=0.0, loading=False, outcome="completed"):
        self.result_ref = result_ref
        self.delay = delay
        self.loading = loading
        self.outcome = outcome
        self.tasks = {}

    def submit(self, **kwargs):
        task = Task(task_id=f"t{len(self.tasks) + 1}", operation=kwargs["operation"],
                    engine=kwargs["engine"], model_id=kwargs.get("model_id") or "",
                    params=kwargs.get("params") or {})
        if self.loading:
            task.state = TaskState.MODEL_LOADING  # the worker is loading the engine
        self.tasks[task.task_id] = task
        return task

    def get(self, task_id):
        return self.tasks.get(task_id)

    def cancel(self, task_id, reason=""):
        return True

    async def wait(self, task_id, timeout=None):
        await asyncio.sleep(self.delay)
        task = self.tasks[task_id]
        attempt = Attempt(attempt_id="a1", task_id=task_id, worker_id=WORKER,
                          session_epoch=1, attempt_number=1)
        attempt.accepted_at = 1.0
        attempt.state = AttemptState.COMMITTED
        task.attempts.append(attempt)
        task.state = TaskState.COMPLETED if self.outcome == "completed" else TaskState.FAILED
        task.result_ref = self.result_ref
        return task


class FakePlane:
    def __init__(self, scheduler, pool=None):
        self.running = True
        self.scheduler = scheduler
        self.pool = pool


def _call(mods, text="Read this line aloud, slowly and warmly.", timed=None, **kw):
    return mods.gateway.RemoteCall(
        engine="omnivoice", operation="tts", params={"text": text},
        timed={"num_step": 32, "speed": 1.0} if timed is None else timed, **kw)


def _run(mods, call, plane, *, job=None):
    async def go():
        return await mods.gateway.run(
            "tts", local=mods.gateway.LocalCall(lambda: "local", what="TTS generate"),
            remote=call, decision=REMOTE, control_plane=plane, job=job)

    return asyncio.run(go())


@pytest.fixture(autouse=False)
def no_preflight(monkeypatch, mods):
    async def _ok(*a, **k):
        return None

    monkeypatch.setattr(mods.gateway, "preflight", _ok)


def test_a_remote_take_is_timed_end_to_end_under_the_workers_bucket(
        db, enrolled, no_preflight, tmp_path, mods):
    artifact = _wav(tmp_path / "take.wav", 2.5)
    plane = FakePlane(FakeScheduler(result_ref=artifact, delay=0.3))
    call = _call(mods)
    waveform, rate = _run(mods, call, plane)
    assert rate == 8000 and waveform.shape[-1] == 20000
    (row,) = _rows(db)
    assert (row["engine"], row["device"], row["num_step"]) == ("omnivoice", DEVICE, 32)
    assert row["text_chars"] == len(call.params["text"])
    assert row["audio_seconds"] == pytest.approx(2.5)
    assert row["wall_seconds"] >= 0.3  # queue + worker + download, not the GPU alone
    assert row["cold"] is None and row["ref_seconds"] is None
    # Numbers and identifiers only: nothing of the text, voice or artifact path.
    assert "Read this" not in json.dumps(row) and "take.wav" not in json.dumps(row)


def test_a_model_load_on_the_worker_marks_the_task_cold(db, enrolled, no_preflight, tmp_path,
                                                        mods):
    artifact = _wav(tmp_path / "take.wav", 1.0)
    _run(mods, _call(mods), FakePlane(FakeScheduler(result_ref=artifact, delay=0.7,
                                                    loading=True)))
    assert [r["cold"] for r in _rows(db)] == ["load"]


def test_a_worker_without_the_engine_resident_is_cold(db, enrolled, no_preflight, tmp_path,
                                                      mods):
    class Capacity:
        def is_resident(self, engine, model_id):
            return False

    live = types.SimpleNamespace(
        capacity=Capacity(),
        record=types.SimpleNamespace(capabilities=[{"engine": "omnivoice",
                                                    "model_id": "omnivoice:default"}]))
    pool = types.SimpleNamespace(get=lambda worker_id: live)
    artifact = _wav(tmp_path / "take.wav", 1.0)
    _run(mods, _call(mods), FakePlane(FakeScheduler(result_ref=artifact), pool=pool))
    assert [r["cold"] for r in _rows(db)] == ["load"]


def test_untimed_and_failed_tasks_record_nothing(db, enrolled, no_preflight, tmp_path, mods):
    artifact = _wav(tmp_path / "take.wav", 1.0)
    untimed = mods.gateway.RemoteCall(engine="omnivoice", operation="tts",
                                      params={"text": "x" * 10})
    _run(mods, untimed, FakePlane(FakeScheduler(result_ref=artifact)))
    job = mods.gateway.JobRun("audiobook")
    assert _run(mods, _call(mods), FakePlane(FakeScheduler(result_ref=artifact,
                                                           outcome="failed")), job=job) == "local"
    assert _rows(db) == []


def test_a_broken_timing_table_never_breaks_the_remote_render(enrolled, no_preflight, tmp_path,
                                                             monkeypatch, mods):
    def broken():
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(mods.core_db, "get_db", broken)
    artifact = _wav(tmp_path / "take.wav", 1.0)
    waveform, _rate = _run(mods, _call(mods), FakePlane(FakeScheduler(result_ref=artifact)))
    assert waveform.shape[-1] == 8000


# ── The estimate reads the bucket the render will go to ──────────────────────


@pytest.fixture
def client(db, monkeypatch, mods):
    from api.routers import render_estimate

    cpu = mods.device_caps.HostCaps(family="cpu", available_families=("cpu",))
    monkeypatch.setattr(mods.device_caps, "detect_host_caps", lambda: cpu)
    tts_backend = importlib.import_module("services.tts_backend")
    monkeypatch.setattr(tts_backend, "active_backend_id", lambda: "kittentts")
    monkeypatch.setattr(importlib.import_module("services.render_warmth"), "engine_is_warm",
                        lambda cls: True)
    app = FastAPI()
    app.include_router(render_estimate.router)
    return TestClient(app)


def _seed(mods, engine, device, num_step, points, cold=None, chars_per_second=15.0):
    for audio, wall in points:
        assert mods.timing.record(engine, device, num_step, int(audio * chars_per_second), audio,
                                  wall, 1.0, None, cold)


_LONG = " ".join(["Every sentence here carries enough words to fill a chunk."] * 40)


def test_a_remote_take_is_priced_from_the_workers_rows_only(client, enrolled, monkeypatch, mods):
    # This machine has plenty of rows; the worker has none yet.
    _seed(mods, "kittentts", "cpu", None, [(x, 1 + 0.5 * x) for x in (2.0, 4.0, 8.0, 16.0)])
    body = client.post("/render/estimate", json={"surface": "generate", "text": _LONG}).json()
    assert body["basis"] == "none" and body["reason"] == "cold_start"
    assert body["device"] == f"remote:{WORKER}:rtx-5080"
    assert body["target"] == {"kind": "remote", "label": "KAIROS"}

    # Worker rows: one whole take per task, 2 s of round trip plus 0.1 s per second.
    _seed(mods, "kittentts", DEVICE, None, [(x, 2 + 0.1 * x) for x in (10.0, 40.0, 90.0, 160.0)])
    body = client.post("/render/estimate", json={"surface": "generate", "text": _LONG}).json()
    assert body["basis"] == "measured" and body["device"] == DEVICE
    # The worker splits the take itself: one task, however many chunks.
    assert body["calls"] == 1 and body["parts"] == [{"calls": 1, "seconds": body["seconds"]}]
    audio = body["audio_seconds"]
    assert body["seconds"] == pytest.approx(2 + 0.1 * audio, rel=0.05)

    # Back on Local: the worker's rows never price a local render.
    monkeypatch.setattr(mods.routing, "get_target_id", lambda: mods.routing.LOCAL)
    local = client.post("/render/estimate", json={"surface": "generate", "text": _LONG}).json()
    assert local["device"] == "cpu" and local["target"] == {"kind": "local"}
    assert local["calls"] > 1


def test_an_offline_worker_is_named_instead_of_estimated(client, enrolled, mods):
    _seed(mods, "kittentts", "cpu", None, [(x, 1 + 0.5 * x) for x in (2.0, 4.0, 8.0)])
    _seed(mods, "kittentts", DEVICE, None, [(x, 2 + 0.1 * x) for x in (10.0, 40.0, 90.0)])
    enrolled["target"] = _target(connected=False, available=False, status="offline",
                                 detail="offline")
    for body in (
        client.post("/render/estimate", json={"surface": "generate", "text": _LONG}).json(),
        client.post("/render/estimate", json={
            "surface": "audiobook", "text": "# One\nFirst.\n# Two\nSecond."}).json(),
    ):
        assert body["basis"] == "none" and body["reason"] == "remote_unavailable"
        assert body["seconds"] is None
        assert body["target"] == {"kind": "unavailable", "label": "KAIROS", "offline": True}


def test_a_remote_book_is_one_task_per_chapter_plus_a_load_when_not_resident(
        client, enrolled, monkeypatch, mods):
    _seed(mods, "kittentts", DEVICE, None, [(x, 2 + 0.1 * x) for x in (10.0, 40.0, 90.0, 200.0)])
    _seed(mods, "kittentts", DEVICE, None, [(20.0, 2 + 0.1 * 20 + 15.0)], cold="load")
    book = "# One\n" + _LONG + "\n# Two\n" + _LONG[:400]
    body = client.post("/render/estimate", json={"surface": "audiobook", "text": book}).json()
    assert body["basis"] == "measured" and [p["calls"] for p in body["parts"]] == [1, 1]
    assert body["warmup_seconds"] == 0.0

    monkeypatch.setattr(mods.remote, "worker_resident", lambda worker_id, engine, **kw: False)
    cold = client.post("/render/estimate", json={"surface": "audiobook", "text": book}).json()
    assert cold["warmup_seconds"] == pytest.approx(15.0, abs=0.5)
    # Paid once, by the first chapter's task.
    assert cold["parts"][0]["seconds"] == pytest.approx(body["parts"][0]["seconds"] + 15.0,
                                                        abs=0.6)
    assert cold["parts"][1]["seconds"] == pytest.approx(body["parts"][1]["seconds"], abs=0.1)


# ── Replay: record through the real gateway, then estimate the next take ─────


def test_replay_a_remote_worker_then_estimate_the_next_take(
        client, db, enrolled, no_preflight, tmp_path, mods):
    """A fake worker 0.05 s of round trip plus 4 ms per audio second: five
    takes recorded through the real gateway and the real recorder, then the
    real endpoint prices a sixth take, which then runs."""
    overhead, per_second, rate = 0.05, 0.004, 15.0

    def take(text):
        seconds = len(text) / rate
        artifact = _wav(tmp_path / f"t{len(text)}.wav", seconds)
        call = mods.gateway.RemoteCall(engine="kittentts", operation="tts",
                                       params={"text": text},
                                       timed={"num_step": None, "speed": 1.0})
        plane = FakePlane(FakeScheduler(result_ref=artifact,
                                        delay=overhead + per_second * seconds))
        started = time.perf_counter()
        _run(mods, call, plane)
        return time.perf_counter() - started

    sentence = "Every sentence here carries enough words to fill a chunk. "
    for n in (2, 4, 8, 12, 16):
        take(sentence * n)
    assert len(_rows(db)) == 5 and {r["device"] for r in _rows(db)} == {DEVICE}

    text = (sentence * 10).strip()
    body = client.post("/render/estimate", json={
        "surface": "generate", "text": text, "max_chunk_chars": 0}).json()
    assert body["basis"] == "measured" and body["samples"] == 5
    actual = take(text)
    assert body["low"] <= actual <= body["high"] + 0.05
    assert body["seconds"] == pytest.approx(actual, rel=0.35)


# ── Live countdown: a remote book re-fits per chapter ────────────────────────


def test_a_remote_book_counts_down_per_chapter(tmp_path, monkeypatch, mods):
    """Drives the real SSE generator with chapters sent to the worker: each
    planned at 10 s, each really taking 20 s, the second one served from cache."""
    from services.ffmpeg_utils import find_ffmpeg

    if find_ffmpeg() is None:
        pytest.skip("ffmpeg required for the longform render")
    audiobook = importlib.import_module("api.routers.audiobook")
    render_estimate = importlib.import_module("api.routers.render_estimate")
    longform_progress = importlib.import_module("services.longform_progress")
    from services.audiobook import AudiobookPlan, Chapter, Span

    clock = {"t": 0.0}
    monkeypatch.setattr(longform_progress, "time",
                        types.SimpleNamespace(perf_counter=lambda: clock["t"]))
    monkeypatch.setattr(mods.gateway, "decide", lambda op, **kw: REMOTE)
    planned = {}

    def plan_longform(plan, **kw):
        planned["route"] = kw.get("route")
        return {"seconds": 40.0, "parts": [
            {"calls": 1, "seconds": 10.0, "call_seconds": [10.0]} for _ in range(4)]}

    monkeypatch.setattr(render_estimate, "plan_longform", plan_longform)
    monkeypatch.setattr(mods.remote, "device_for", lambda worker_id, **kw: DEVICE)
    wav = _wav(tmp_path / "chapter.wav", 0.5, rate=24000)

    async def run_chapter(chapter, **kw):
        assert kw.get("on_call") is None  # a worker reports nothing inside a chapter
        cached = chapter.title == "Two"
        clock["t"] += 0.0 if cached else 20.0
        return wav, 0.5, cached, None

    monkeypatch.setattr(audiobook, "_run_chapter", run_chapter)
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    plan = AudiobookPlan(chapters=[Chapter(title=t, spans=[Span(voice_id=None, text="Hi.")])
                                   for t in ("One", "Two", "Three", "Four")])

    async def run():
        return [json.loads(f[len("data:"):]) async for f in
                audiobook._render_longform_sse(plan, default_voice=None, fmt="mp3")]

    events = asyncio.run(asyncio.wait_for(run(), timeout=120))
    assert planned["route"].kind == "remote" and planned["route"].device == DEVICE
    progress = [e for e in events if e["type"] == "progress"]
    # Start; after One (20 s, pace needs two chapters); after Two (cached: free,
    # teaches nothing); after Three (pace 2.0: Four left at 20 s); after Four.
    assert [p["remaining_s"] for p in progress] == [40.0, 30.0, 20.0, 20.0, 0.0]
    assert progress[3]["pace"] == 2.0 and progress[3]["next_call_s"] == 20.0
    assert [e["type"] for e in events if e["type"] == "chapter"] == ["chapter"] * 4
