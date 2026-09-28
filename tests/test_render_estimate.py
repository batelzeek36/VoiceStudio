"""POST /render/estimate: plans the calls the real render makes, prices them
with this machine's own timings, and stays silent until it has three samples
(docs/adr/render-time-estimate.md)."""
import importlib
import sqlite3

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.db as core_db
from core import device_caps
from services import render_plan, render_timing

_CPU = device_caps.HostCaps(family="cpu", available_families=("cpu",))


@pytest.fixture(autouse=True)
def _live_backend_modules():
    """Rebind this file's module aliases to the live modules before each test.

    tests/backend/conftest.py purges ``core.*`` and ``services.*`` from
    sys.modules after each of its tests, so in a full-suite run an alias bound
    at collection is a stale object, while the code under test imports lazily
    and never sees a patch made on the stale one (tests/conftest.py,
    asr_model_installed, has the same note)."""
    global core_db, device_caps, render_plan, render_timing, _CPU
    core_db = importlib.import_module("core.db")
    device_caps = importlib.import_module("core.device_caps")
    render_plan = importlib.import_module("services.render_plan")
    render_timing = importlib.import_module("services.render_timing")
    _CPU = device_caps.HostCaps(family="cpu", available_families=("cpu",))


def _connect(path):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "estimate.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(core_db._BASE_SCHEMA)
    monkeypatch.setattr(core_db, "get_db", lambda: _connect(path))
    return path


@pytest.fixture
def client(db, monkeypatch):
    """The estimate router on a bare app, on a CPU host, never remote, and
    with every model loader booby-trapped: an estimate must not load one."""
    from api.routers import render_estimate
    from services import gpu_gateway, model_manager, tts_backend

    monkeypatch.setattr(device_caps, "detect_host_caps", lambda: _CPU)
    monkeypatch.setattr(tts_backend, "active_backend_id", lambda: "kittentts")

    class _Local:
        remote = False

    monkeypatch.setattr(gpu_gateway, "decide", lambda op, **kw: _Local())
    # The GPU picker says Local unless a test says otherwise (patched on the
    # router's own module object, which is the one it calls).
    monkeypatch.setattr(render_estimate.render_remote, "route_for",
                        lambda op, **kw: render_estimate.render_remote.LOCAL_ROUTE)

    async def no_model(*_a, **_k):
        raise AssertionError("an estimate must never load a model")

    monkeypatch.setattr(model_manager, "get_model", no_model)
    # The engine is loaded and has rendered here; tests of cold starts flip it.
    from services import render_warmth

    monkeypatch.setattr(render_warmth, "engine_is_warm", lambda cls: True)
    monkeypatch.setattr(tts_backend.KittenTTSBackend, "generate",
                        lambda *a, **k: pytest.fail("an estimate must never synthesize"))
    app = FastAPI()
    app.include_router(render_estimate.router)
    return TestClient(app)


def _seed(engine, device, num_step, points, speed=1.0, chars_per_second=15.0,
          ref_seconds=None, cold=None):
    for audio, wall in points:
        assert render_timing.record(engine, device, num_step,
                                    int(audio * chars_per_second), audio, wall, speed,
                                    ref_seconds, cold)


_LONG = " ".join(["Every sentence here carries enough words to fill a chunk."] * 40)


# ── Call planning mirrors the renders ─────────────────────────────────────────


class _Engine:
    sample_rate = 1000
    applies_own_mastering = True

    def __init__(self):
        self.texts = []

    def generate(self, text, **kwargs):
        self.texts.append(text)
        return torch.zeros(1, 100)


@pytest.mark.parametrize("text,max_chars", [
    ("A short line.", 800),
    (_LONG, 800),
    (_LONG, 0),
    ("One. [pause 400ms] Two. [pause 1s] Three.", 800),
    ("[pause 500ms]", 800),
    ("Only [pause 200ms] pauses [pause 200ms]", 120),
])
def test_generate_plan_matches_the_calls_the_render_makes(text, max_chars):
    from api.routers.generation import _run_backend_inference

    engine = _Engine()
    _run_backend_inference(engine, text, None, None, None, None, None, 16, 2.0, 1.0,
                           False, False, 1, "raw", max_chars, 0)
    planned = render_plan.generate_calls(text, max_chunk_chars=max_chars)
    assert [c.text for c in planned] == engine.texts


def test_generate_plan_keeps_duration_only_for_a_single_call():
    assert render_plan.generate_calls("Hello.", max_chunk_chars=800, duration=4.0)[0].duration == 4.0
    chunked = render_plan.generate_calls(_LONG, max_chunk_chars=800, duration=4.0)
    assert len(chunked) > 1 and all(c.duration is None for c in chunked)
    assert render_plan.generate_calls("   ", max_chunk_chars=800) == []


_SCRIPT = """# One
Plain narration that runs on.
[voice:Mara] She answers. [pause 600ms] Then [slow]slowly[/slow] continues.

A new paragraph by the same voice. [spell]ABC[/spell]

# Two
[voice:Cole] Tomato and tomato again.
"""


@pytest.mark.parametrize("paragraph_gap_ms", [0, 350])
def test_chapter_plan_matches_the_calls_synthesize_chapter_makes(paragraph_gap_ms):
    from services.audiobook import normalized_spans, parse_audiobook_script, synthesize_chapter

    lexicon = {"tomato": "tuh-MAH-toe"}
    plan = parse_audiobook_script(_SCRIPT, default_voice="narrator")
    for chapter in plan.chapters:
        spans = normalized_spans(chapter.spans, "English")
        seen = []

        def synth(text, voice, speed):
            seen.append((text, voice, speed or 1.0))
            return torch.zeros(1, 50)

        synthesize_chapter(spans, synth, 1000, crossfade_ms=0, lexicon=lexicon,
                           paragraph_gap_ms=paragraph_gap_ms)
        planned = render_plan.chapter_calls(spans, lexicon=lexicon,
                                            paragraph_gap_ms=paragraph_gap_ms)
        assert [(c.text, c.voice, c.speed) for c in planned] == seen
    # Voice switches, the pause and the markup split are separate calls.
    first = render_plan.chapter_calls(normalized_spans(plan.chapters[0].spans, "English"),
                                      lexicon=None, paragraph_gap_ms=paragraph_gap_ms)
    assert len(first) >= 5
    assert {c.voice for c in first} == {"narrator", "Mara"}


def test_omnivoice_audio_matches_the_models_own_estimator():
    from omnivoice.utils.duration import RuleDurationEstimator

    pace = render_plan.VoicePace(True, "This is my reference transcript.", 3.2)
    text = "A sentence the model has never seen before, of moderate length."
    tokens = RuleDurationEstimator().estimate_duration(
        text, pace.ref_text, int(round(3.2 * render_plan.OMNIVOICE_FRAME_RATE)))
    assert render_plan.omnivoice_seconds(text, pace, 1.0) == pytest.approx(int(tokens) / 25)
    # Speed shortens, and a voice without a transcript uses the model's default.
    assert render_plan.omnivoice_seconds(text, pace, 2.0) < render_plan.omnivoice_seconds(text, pace, 1.0)
    assert render_plan.omnivoice_seconds(text, None) > 0


# ── The endpoint ─────────────────────────────────────────────────────────────


def test_fresh_machine_gets_no_number(client):
    body = client.post("/render/estimate", json={"surface": "generate", "text": _LONG}).json()
    assert body["basis"] == "none" and body["reason"] == "cold_start"
    assert body["seconds"] is None and body["low"] is None and body["high"] is None
    assert body["samples"] == 0
    assert body["calls"] > 1
    assert (body["engine"], body["device"], body["num_step"]) == ("kittentts", "cpu", None)


def test_two_samples_are_still_cold_start(client):
    _seed("kittentts", "cpu", None, [(4.0, 2.0), (8.0, 3.0)])
    body = client.post("/render/estimate", json={"surface": "generate", "text": "Hi there."}).json()
    assert body["basis"] == "none" and body["samples"] == 2


def test_three_samples_give_a_measured_estimate(client):
    # 1 s per call + 0.5 s per audio second; 15 chars per audio second.
    _seed("kittentts", "cpu", None, [(x, 1 + 0.5 * x) for x in (2.0, 4.0, 8.0, 16.0, 24.0)])
    text = "x" * 300  # one call, 20 s of audio at the learned rate
    body = client.post("/render/estimate", json={"surface": "generate", "text": text,
                                                 "max_chunk_chars": 0}).json()
    assert body["basis"] == "measured" and body["reason"] is None
    assert body["calls"] == 1
    assert body["audio_seconds"] == pytest.approx(20.0, abs=0.1)
    assert body["seconds"] == pytest.approx(1 + 0.5 * 20, abs=0.2)
    assert body["low"] <= body["seconds"] <= body["high"]
    assert body["parts"] == [{"calls": 1, "seconds": body["seconds"]}]


def test_each_call_pays_the_per_call_cost(client):
    """Markup-heavy text is many short calls: costed per call, not per character."""
    _seed("kittentts", "cpu", None, [(x, 3 + 0.1 * x) for x in (1.0, 2.0, 4.0, 8.0)])
    one = client.post("/render/estimate", json={
        "surface": "generate", "text": "Word " * 20}).json()
    many = client.post("/render/estimate", json={
        "surface": "generate", "text": " [pause 100ms] ".join(["Word " * 2] * 10)}).json()
    assert many["calls"] == 10 and one["calls"] == 1
    assert many["seconds"] > one["seconds"] + 20


def test_other_steps_give_a_rough_estimate(client, monkeypatch):
    from services import tts_backend

    monkeypatch.setattr(tts_backend, "active_backend_id", lambda: "omnivoice")
    monkeypatch.setattr(tts_backend, "get_backend_class",
                        lambda engine_id: tts_backend.OmniVoiceBackend)
    _seed("omnivoice", "cpu", 16, [(x, 2 + 3 * x) for x in (2.0, 4.0, 6.0)])
    at16 = client.post("/render/estimate", json={
        "surface": "generate", "text": "Hello from the estimate.", "num_step": 16}).json()
    at32 = client.post("/render/estimate", json={
        "surface": "generate", "text": "Hello from the estimate.", "num_step": 32}).json()
    assert at16["basis"] == "measured" and at32["basis"] == "rough"
    assert at32["num_step"] == 32
    audio = at16["audio_seconds"]
    assert at16["seconds"] == pytest.approx(2 + 3 * audio, abs=0.2)
    assert at32["seconds"] == pytest.approx(2 + 6 * audio, abs=0.3)


def test_saved_voice_pace_drives_omnivoice_audio(client, db, tmp_path, monkeypatch):
    from services import tts_backend

    monkeypatch.setattr(tts_backend, "get_backend_class",
                        lambda engine_id: tts_backend.OmniVoiceBackend)
    ref = tmp_path / "ref.wav"
    sf.write(ref, np.zeros(24000 * 4, dtype=np.float32), 24000)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO voice_profiles (id, name, ref_audio_path, ref_text, kind) "
            "VALUES ('v1', 'Slow', ?, 'Two words.', 'clone')", (str(ref),))
    body = {"surface": "generate", "engine": "omnivoice", "text": "One two three four five six."}
    slow = client.post("/render/estimate", json={**body, "profile_id": "v1"}).json()
    default = client.post("/render/estimate", json=body).json()
    # Two words in four seconds is a slow speaker: far more audio.
    assert slow["audio_seconds"] > 2 * default["audio_seconds"]


def test_uploaded_reference_pace_is_used(client, monkeypatch):
    from services import tts_backend

    monkeypatch.setattr(tts_backend, "get_backend_class",
                        lambda engine_id: tts_backend.OmniVoiceBackend)
    body = {"surface": "generate", "engine": "omnivoice", "text": "One two three four five."}
    fast = client.post("/render/estimate", json={
        **body, "ref_text": "A quick reference transcript said fast.", "ref_seconds": 1.5}).json()
    slow = client.post("/render/estimate", json={
        **body, "ref_text": "A quick reference transcript said fast.", "ref_seconds": 6.0}).json()
    assert slow["audio_seconds"] > fast["audio_seconds"]


def test_audiobook_parts_follow_the_chapters(client):
    _seed("kittentts", "cpu", None, [(x, 1 + 0.5 * x) for x in (0.2, 2.0, 4.0, 8.0)])
    body = client.post("/render/estimate", json={
        "surface": "audiobook", "text": _SCRIPT, "default_voice": None}).json()
    assert body["basis"] == "measured"
    assert len(body["parts"]) == 2
    assert sum(p["calls"] for p in body["parts"]) == body["calls"]
    assert sum(p["seconds"] for p in body["parts"]) == pytest.approx(body["seconds"], abs=0.2)


def test_stories_plan_drops_empty_chapters_like_the_render(client):
    body = client.post("/render/estimate", json={
        "surface": "longform",
        "chapters": [
            {"title": "Empty", "spans": [{"text": "   "}]},
            {"title": "Real", "spans": [{"text": "A line.", "speed": 1.2},
                                        {"text": "", "pause_ms_after": 500}]},
        ],
    }).json()
    assert len(body["parts"]) == 1 and body["calls"] == 1
    assert client.post("/render/estimate", json={"surface": "longform"}).status_code == 422


def test_unknown_engine_and_bad_numbers_are_rejected(client):
    assert client.post("/render/estimate", json={
        "surface": "generate", "text": "x", "engine": "nope"}).status_code == 400
    assert client.post("/render/estimate", json={
        "surface": "generate", "text": "x", "speed": 0}).status_code == 422
    assert client.post("/render/estimate", json={"surface": "nope"}).status_code == 422


def test_estimates_are_identical_across_devices_only_in_behaviour(client, monkeypatch):
    """Buckets separate devices: CUDA timings never price a CPU render."""
    _seed("kittentts", "cuda", None, [(x, 0.1 * x) for x in (2.0, 4.0, 8.0)])
    body = client.post("/render/estimate", json={"surface": "generate", "text": "Hi."}).json()
    assert body["device"] == "cpu" and body["basis"] == "none"


# ── Cold starts: added only when the render will pay them ─────────────────────


def test_an_unloaded_engine_adds_the_measured_load_cost(client, monkeypatch):
    from services import render_warmth

    _seed("kittentts", "cpu", None, [(x, 1 + 0.5 * x) for x in (2.0, 4.0, 8.0)])
    _seed("kittentts", "cpu", None, [(4.0, 3.0 + 9.0)], cold="load")  # 9 s of load
    body = {"surface": "generate", "text": "x" * 120, "max_chunk_chars": 0}
    warm = client.post("/render/estimate", json=body).json()
    monkeypatch.setattr(render_warmth, "engine_is_warm", lambda cls: False)
    cold = client.post("/render/estimate", json=body).json()
    assert warm["basis"] == cold["basis"] == "measured"
    assert warm["samples"] == cold["samples"] == 3  # the cold call is not in the line
    assert warm["warmup_seconds"] == 0
    assert cold["warmup_seconds"] == pytest.approx(9.0, abs=0.2)
    assert cold["seconds"] == pytest.approx(warm["seconds"] + 9.0, abs=0.2)
    assert cold["parts"][0]["seconds"] == cold["seconds"]


def test_an_unmeasured_load_cost_widens_the_range_instead(client, monkeypatch):
    from services import render_warmth

    _seed("kittentts", "cpu", None, [(x, 1 + 0.5 * x) for x in (2.0, 4.0, 8.0)])
    body = {"surface": "generate", "text": "x" * 120, "max_chunk_chars": 0}
    warm = client.post("/render/estimate", json=body).json()
    monkeypatch.setattr(render_warmth, "engine_is_warm", lambda cls: False)
    cold = client.post("/render/estimate", json=body).json()
    assert cold["basis"] == "rough"
    assert cold["seconds"] == warm["seconds"]
    assert cold["high"] > warm["high"] + 0.9 * warm["seconds"]


def test_a_long_unranked_reference_adds_the_voice_cost_once(client, db, tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from services import tts_backend

    monkeypatch.setattr(tts_backend, "active_backend_id", lambda: "omnivoice")
    monkeypatch.setattr(tts_backend, "get_backend_class",
                        lambda engine_id: tts_backend.OmniVoiceBackend)
    monkeypatch.setattr(tts_backend, "_recall_passage", lambda path: None)
    ref = tmp_path / "long.wav"
    sf.write(ref, np.zeros(24000 * 40, dtype=np.float32), 24000)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO voice_profiles (id, name, ref_audio_path, ref_text, kind) "
                     "VALUES ('long', 'Long', ?, 'A long reference transcript.', 'clone')",
                     (str(ref),))
    _seed("omnivoice", "cpu", 32, [(x, 2 + 0.4 * (x + 15)) for x in (3.0, 6.0, 12.0)],
          ref_seconds=15.0)
    _seed("omnivoice", "cpu", 32, [(6.0, 2 + 0.4 * 21 + 20.0)], ref_seconds=15.0, cold="voice")
    body = client.post("/render/estimate", json={
        "surface": "audiobook", "text": "# One\nFirst chapter.\n# Two\nSecond chapter.",
        "default_voice": "long", "num_step": 32}).json()
    assert body["warmup_seconds"] == pytest.approx(20.0, abs=0.3)
    first, second = body["parts"]
    # Paid once, in the chapter where the voice is first heard.
    assert first["seconds"] > second["seconds"] + 19


def test_a_call_longer_than_anything_measured_is_rough(client):
    _seed("kittentts", "cpu", None, [(x, 1 + 0.5 * x) for x in (2.0, 4.0, 8.0)])
    inside = client.post("/render/estimate", json={
        "surface": "generate", "text": "x" * 90, "max_chunk_chars": 0}).json()
    beyond = client.post("/render/estimate", json={
        "surface": "generate", "text": "x" * 600, "max_chunk_chars": 0}).json()
    assert inside["basis"] == "measured" and beyond["basis"] == "rough"
    assert beyond["high"] / beyond["seconds"] > inside["high"] / inside["seconds"]
