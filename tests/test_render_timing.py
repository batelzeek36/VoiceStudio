"""Per-call render timings: recorded at every synthesis chokepoint, content-free,
capped, and never able to break a render (docs/adr/render-time-estimate.md)."""
import importlib
import json
import sqlite3
from pathlib import Path

import pytest
import torch

import core.db as core_db
from services import render_timing


@pytest.fixture(autouse=True)
def _live_backend_modules():
    """Rebind this file's module aliases to the live modules before each test.

    tests/backend/conftest.py purges ``core.*`` and ``services.*`` from
    sys.modules after each of its tests, so in a full-suite run an alias bound
    at collection is a stale object, while the code under test imports lazily
    and never sees a patch made on the stale one (tests/conftest.py,
    asr_model_installed, has the same note)."""
    global core_db, render_timing
    core_db = importlib.import_module("core.db")
    render_timing = importlib.import_module("services.render_timing")


def _connect(path):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture
def timing_db(tmp_path, monkeypatch):
    """Every core.db connection goes to a throwaway, fully initialised file."""
    path = tmp_path / "timings.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(core_db._BASE_SCHEMA)
    monkeypatch.setattr(core_db, "get_db", lambda: _connect(path))
    return path


def _rows(path):
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM render_timings ORDER BY id")]


class _Engine:
    sample_rate = 1000

    def __init__(self, samples=500):
        self.samples = samples
        self.texts = []

    def generate(self, text, **kwargs):
        self.texts.append(text)
        return torch.zeros(1, self.samples)


def _timing(engine="omnivoice", device="mps", num_step=16, sample_rate=1000):
    return render_timing.SynthesisTiming(engine, device, num_step, sample_rate)


def test_a_successful_call_records_numbers_only(timing_db):
    engine = _Engine(samples=2500)
    out = _timing().call("A private sentence.", 1.25, None, engine.generate,
                         "A private sentence.", speed=1.25)
    assert out.shape[-1] == 2500
    row, = _rows(timing_db)
    assert row["engine"] == "omnivoice" and row["device"] == "mps" and row["num_step"] == 16
    assert row["text_chars"] == len("A private sentence.")
    assert row["audio_seconds"] == pytest.approx(2.5)
    assert row["wall_seconds"] > 0
    assert row["speed"] == pytest.approx(1.25)
    # No text, voice or path is stored anywhere in the row.
    assert "private" not in json.dumps(row)
    columns = set(row)
    assert columns == {"id", "engine", "device", "num_step", "text_chars", "audio_seconds",
                       "wall_seconds", "speed", "ref_seconds", "cold", "created_at"}
    assert row["ref_seconds"] is None and row["cold"] is None


def test_failed_and_empty_calls_record_nothing(timing_db):
    def boom(*_a, **_k):
        raise RuntimeError("engine fell over")

    with pytest.raises(RuntimeError):
        _timing().call("text", None, None, boom)
    _timing().call("text", None, None, _Engine(samples=0).generate, "text")
    assert _rows(timing_db) == []


def test_list_outputs_and_lazy_sample_rates(timing_db):
    """generate_with_cached_ref returns a list; lazy engines learn their rate late."""
    rate = {"sr": 8000}
    timing = _timing(sample_rate=lambda: rate["sr"])
    rate["sr"] = 24000
    timing.call("text", None, None, lambda: [torch.zeros(1, 48000)])
    assert _rows(timing_db)[0]["audio_seconds"] == pytest.approx(2.0)


def test_a_broken_database_never_breaks_the_render(monkeypatch):
    def broken():
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(core_db, "get_db", broken)
    engine = _Engine()
    assert _timing().call("text", None, None, engine.generate, "text").shape[-1] == 500
    assert render_timing.record("omnivoice", "cpu", None, 4, 1.0, 1.0) is False


def test_missing_table_self_heals_and_records(tmp_path, monkeypatch):
    path = tmp_path / "bare.db"
    sqlite3.connect(path).close()
    monkeypatch.setattr(core_db, "get_db", lambda: _connect(path))
    assert render_timing.record("omnivoice", "cpu", None, 4, 1.0, 2.0) is True
    assert len(_rows(path)) == 1


def test_rows_are_capped_per_bucket_and_in_total(timing_db, monkeypatch):
    monkeypatch.setattr(render_timing, "MAX_ROWS_PER_BUCKET", 5)
    monkeypatch.setattr(render_timing, "MAX_ROWS_TOTAL", 8)
    for i in range(9):
        render_timing.record("omnivoice", "mps", 16, 10, 1.0 + i, 2.0)
    for i in range(3):
        render_timing.record("omnivoice", "mps", 32, 10, 1.0, 2.0)
    rows = _rows(timing_db)
    at16 = [r for r in rows if r["num_step"] == 16]
    assert len(at16) == 5
    # The newest survive.
    assert [r["audio_seconds"] for r in at16] == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert len(rows) == 8


def test_invalid_numbers_are_rejected(timing_db):
    assert not render_timing.record("omnivoice", "mps", 16, 10, float("nan"), 1.0)
    assert not render_timing.record("omnivoice", "mps", 16, 10, 1.0, -1.0)
    assert not render_timing.record("omnivoice", "mps", 16, 0, 1.0, 1.0)
    assert not render_timing.record("", "mps", 16, 10, 1.0, 1.0)
    assert _rows(timing_db) == []


def test_reads_return_newest_first_and_learn_speech_rate(timing_db):
    for i in range(4):
        render_timing.record("kittentts", "cpu", None, 100, 5.0, 1.0 + i, 1.0)
    render_timing.record("kittentts", "cpu", None, 100, 4.0, 9.0, 1.25)
    samples = render_timing.samples_for("kittentts", "cpu", None)
    assert [s.wall_seconds for s in samples][:2] == [9.0, 4.0]
    # 4 x 5 s + 4 s at 1.25x (= 5 s at 1x) over 500 chars.
    assert render_timing.seconds_per_char("kittentts") == pytest.approx(25.0 / 500)
    assert render_timing.seconds_per_char("nobody") is None


def test_steps_bucket_only_for_engines_that_read_steps():
    from services.tts_backend import KittenTTSBackend, OmniVoiceBackend

    assert render_timing.timing_steps(OmniVoiceBackend, 32) == 32
    assert render_timing.timing_steps(OmniVoiceBackend, None) is None
    assert render_timing.timing_steps(KittenTTSBackend, 16) is None


def test_device_class_is_the_routing_answer_and_names_directml(monkeypatch):
    from core import device_caps
    from services.tts_backend import KittenTTSBackend, OmniVoiceBackend

    caps = device_caps.HostCaps(
        family="cpu", available_families=("cpu",),
        notes=(f"{device_caps.DIRECTML_MARKER} (Windows GPU); x",),
    )
    monkeypatch.setattr(device_caps, "detect_host_caps", lambda: caps)
    assert render_timing.device_class(OmniVoiceBackend) == "directml"
    assert render_timing.device_class(KittenTTSBackend) == "cpu"
    cuda = device_caps.HostCaps(family="cuda", available_families=("cuda", "cpu"))
    monkeypatch.setattr(device_caps, "detect_host_caps", lambda: cuda)
    assert render_timing.device_class(OmniVoiceBackend) == "cuda"
    # A CPU-only engine on a GPU host is timed as CPU work.
    assert render_timing.device_class(KittenTTSBackend) == "cpu"


def test_engine_key_names_the_model_behind_a_multi_model_adapter(monkeypatch):
    from services.tts_backend import MLXAudioBackend, OmniVoiceBackend

    monkeypatch.setenv("OMNIVOICE_MLX_AUDIO_MODEL", "mlx-community/Kokoro-82M-bf16")
    assert render_timing.engine_key("mlx-audio", MLXAudioBackend) == \
        "mlx-audio:mlx-community-kokoro-82m-bf16"
    assert render_timing.engine_key("omnivoice", OmniVoiceBackend) == "omnivoice"


def test_for_render_never_raises():
    class Weird:
        @classmethod
        def runtime_compute_profile(cls, caps):
            raise RuntimeError("probe failed")

    timing = render_timing.for_render("weird", Weird, num_step=8, sample_rate=1000)
    assert timing.device  # falls back to the host family
    assert timing.num_step is None  # Weird does not read steps


def test_recording_stays_out_of_analytics_and_bug_reports():
    """ADR: the rows never leave the machine."""
    root = Path(__file__).resolve().parents[1]
    # v0.5.6: the renderer's analytics live in frontend/src/utils/analytics.ts
    # (electron/src/shared/utils/analytics.ts upstream).
    for rel in ("backend/core/analytics.py", "backend/core/diagnostic_bundle.py",
                "frontend/src/utils/analytics.ts"):
        assert "render_timing" not in (root / rel).read_text(encoding="utf-8"), rel


# ── Every synthesis chokepoint records ────────────────────────────────────────


def test_generate_backend_path_records_every_chunk(timing_db):
    from api.routers.generation import _run_backend_inference

    engine = _Engine()
    engine.applies_own_mastering = True
    text = " ".join(["This sentence is long enough to be its own chunk."] * 6)
    _run_backend_inference(
        engine, text, None, None, None, None, None, 16, 2.0, 1.0, False, False, 1,
        "raw", 60, 0, timing=_timing(sample_rate=lambda: engine.sample_rate),
    )
    rows = _rows(timing_db)
    assert len(rows) == len(engine.texts) > 1
    assert [r["text_chars"] for r in rows] == [len(t) for t in engine.texts]


def test_generate_pause_path_records_each_span(timing_db):
    from api.routers.generation import _run_backend_inference

    engine = _Engine()
    engine.applies_own_mastering = True
    _run_backend_inference(
        engine, "First. [pause 300ms] Second.", None, None, None, None, None, 16,
        2.0, 1.0, False, False, 1, "raw", timing=_timing(),
    )
    assert len(_rows(timing_db)) == 2 == len(engine.texts)


def test_native_generate_path_records(timing_db, monkeypatch):
    from api.routers import generation
    from services import tts_backend

    class Model:
        sampling_rate = 1000

        def generate(self, **kwargs):
            return [torch.zeros(1, 700)]

    monkeypatch.setattr(tts_backend, "generate_with_cached_ref",
                        lambda model, **kw: model.generate(**kw))
    generation._run_inference(
        Model(), "Hello there.", None, None, None, None, None, 16, 2.0, 1.0,
        None, False, False, None, None, None, None, "raw", timing=_timing(),
    )
    row, = _rows(timing_db)
    assert row["audio_seconds"] == pytest.approx(0.7)


def test_chapter_render_records_each_chunk(timing_db):
    from services.audiobook import Span, synthesize_chapter

    spans = [Span(voice_id=None, text="One line."), Span(voice_id="b", text="Two line.",
                                                         speed=1.5)]
    synthesize_chapter(spans, lambda text, voice, speed: torch.zeros(1, 1000), 1000,
                       crossfade_ms=0, timing=_timing(engine="kittentts", device="cpu",
                                                      num_step=None))
    rows = _rows(timing_db)
    assert [r["speed"] for r in rows] == [None, 1.5]
    assert all(r["engine"] == "kittentts" for r in rows)


def test_untimed_paths_record_nothing(timing_db):
    from services.audiobook import Span, synthesize_chapter

    synthesize_chapter([Span(voice_id=None, text="One line.")],
                       lambda *a: torch.zeros(1, 1000), 1000, crossfade_ms=0)
    assert _rows(timing_db) == []


# ── Cold calls: flagged from the engine's real state ──────────────────────────


@pytest.fixture
def fresh_warmth():
    from services import render_warmth

    render_warmth._reset_for_tests()
    yield render_warmth
    render_warmth._reset_for_tests()


class _Proc:
    def __init__(self):
        self.code = None

    def poll(self):
        return self.code


def _sidecar_engine():
    """A stand-in for a subprocess engine: its sidecar is ``_proc``."""
    from services.tts_backend import TTSBackend

    class Sidecar(TTSBackend):
        id = "sidecar-test"
        sample_rate = 1000
        supported_languages = ["multi"]

        def __init__(self):
            self._proc = None

        @classmethod
        def is_available(cls):
            return True, "ready"

        def generate(self, text, **kwargs):
            if self._proc is None:
                self._proc = _Proc()  # spawns and loads on the first synthesize
            return torch.zeros(1, 1000)

    return Sidecar


def test_first_call_on_a_fresh_sidecar_is_cold_then_warm(timing_db, fresh_warmth):
    cls = _sidecar_engine()
    engine = cls()
    timing = render_timing.SynthesisTiming("sidecar-test", "mps", None, 1000,
                                           backend_cls=cls, runtime=engine)
    for _ in range(2):
        timing.call("text", None, None, engine.generate, "text")
    assert [r["cold"] for r in _rows(timing_db)] == ["load", None]
    assert fresh_warmth.is_warm(engine)


def test_a_respawned_or_dead_sidecar_is_cold_again(timing_db, fresh_warmth):
    cls = _sidecar_engine()
    engine = cls()
    timing = render_timing.SynthesisTiming("sidecar-test", "mps", None, 1000,
                                           backend_cls=cls, runtime=engine)
    timing.call("text", None, None, engine.generate, "text")
    engine._proc.code = 0  # reaped for idleness / crashed
    assert not fresh_warmth.is_warm(engine)
    engine._proc = None
    timing.call("text", None, None, engine.generate, "text")
    assert [r["cold"] for r in _rows(timing_db)] == ["load", "load"]


def test_a_reloaded_native_model_is_cold(timing_db, fresh_warmth):
    class Model:
        pass

    first, second = Model(), Model()
    for model in (first, first, second):
        render_timing.SynthesisTiming("omnivoice", "cuda", 16, 1000, runtime=model).call(
            "text", None, None, lambda: torch.zeros(1, 1000))
    assert [r["cold"] for r in _rows(timing_db)] == ["load", None, "load"]


@pytest.fixture
def empty_prompt_cache(monkeypatch):
    """No encoded clone prompt in memory or on disk (v0.5.6 keeps a long
    reference's chosen passage only inside its cached prompt)."""
    from collections import OrderedDict

    from services import tts_backend

    monkeypatch.setattr(tts_backend, "_prompt_cache", OrderedDict())
    monkeypatch.setenv("OMNIVOICE_PROMPT_DISK_CACHE", "0")
    return tts_backend


def _refs(tmp_path):
    import numpy as np
    import soundfile as sf

    long_ref = tmp_path / "long.wav"
    sf.write(long_ref, np.zeros(24000 * 40, dtype=np.float32), 24000)
    short_ref = tmp_path / "short.wav"
    sf.write(short_ref, np.zeros(24000 * 9, dtype=np.float32), 24000)
    return str(long_ref), str(short_ref)


def test_first_call_on_an_uncached_long_reference_is_voice_cold(
        timing_db, fresh_warmth, tmp_path, empty_prompt_cache):
    tts_backend = empty_prompt_cache
    long_ref, short_ref = _refs(tmp_path)
    timing = render_timing.SynthesisTiming("omnivoice", "cuda", 16, 1000,
                                           backend_cls=tts_backend.OmniVoiceBackend)

    def synth(ref):
        # The call encodes the prompt (ranking a long reference's windows) and
        # caches it, exactly as tts_backend._get_clone_prompt does.
        tts_backend._prompt_cache[tts_backend._clone_prompt_key(ref, None, True)] = object()
        return torch.zeros(1, 1000)

    for ref in (long_ref, long_ref, short_ref):
        timing.call("text", None, ref, synth, ref)
    rows = _rows(timing_db)
    assert [r["cold"] for r in rows] == ["voice", None, None]
    # A long reference is conditioned on its 15 s window; a short one whole.
    assert [r["ref_seconds"] for r in rows] == [15.0, 15.0, pytest.approx(9.0)]


def test_a_prompt_saved_on_disk_counts_as_a_chosen_passage(
        fresh_warmth, tmp_path, empty_prompt_cache, monkeypatch):
    """After a restart the disk layer serves the prompt: no ranking, no cold."""
    import os

    from core.config import DATA_DIR

    tts_backend = empty_prompt_cache
    monkeypatch.setenv("OMNIVOICE_PROMPT_DISK_CACHE", "1")
    long_ref, _ = _refs(tmp_path)
    cache_dir = os.path.join(str(DATA_DIR), "prompt_cache")
    path = tts_backend._prompt_disk_path(cache_dir, tts_backend._clone_prompt_key(long_ref, None, True))
    if os.path.exists(path):
        os.remove(path)
    assert fresh_warmth.voice_needs_passage(tts_backend.OmniVoiceBackend, long_ref)
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"prompt")
    try:
        assert not fresh_warmth.voice_needs_passage(tts_backend.OmniVoiceBackend, long_ref)
    finally:
        os.remove(path)


def test_checking_the_passage_never_creates_the_cache_directory(
        fresh_warmth, tmp_path, empty_prompt_cache, monkeypatch):
    from core import config

    tts_backend = empty_prompt_cache
    monkeypatch.setenv("OMNIVOICE_PROMPT_DISK_CACHE", "1")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    long_ref, _ = _refs(tmp_path)
    assert fresh_warmth.voice_needs_passage(tts_backend.OmniVoiceBackend, long_ref)
    assert not (tmp_path / "data").exists()


def test_every_sidecar_call_on_a_long_reference_ranks_again(
        timing_db, fresh_warmth, tmp_path, empty_prompt_cache):
    """v0.5.6's OmniVoice sidecar (the effective omnivoice on MPS) hands the
    whole reference to the child's model.generate(), which has no prompt cache."""
    from engines.omnivoice_subprocess import OmniVoiceMPSSubprocessBackend

    long_ref, short_ref = _refs(tmp_path)
    assert fresh_warmth.ranks_every_call(OmniVoiceMPSSubprocessBackend)
    assert not fresh_warmth.ranks_every_call(empty_prompt_cache.OmniVoiceBackend)
    timing = render_timing.SynthesisTiming("omnivoice", "mps", 64, 1000,
                                           backend_cls=OmniVoiceMPSSubprocessBackend)
    for ref in (long_ref, long_ref, short_ref):
        timing.call("text", None, ref, lambda: torch.zeros(1, 1000))
    assert [r["cold"] for r in _rows(timing_db)] == ["voice", "voice", None]


def test_longform_voice_tokens_map_to_their_reference(timing_db, fresh_warmth, tmp_path):
    import numpy as np
    import soundfile as sf
    from services.audiobook import Span, synthesize_chapter
    from services.tts_backend import OmniVoiceBackend

    ref = tmp_path / "mara.wav"
    sf.write(ref, np.zeros(24000 * 12, dtype=np.float32), 24000)
    timing = render_timing.SynthesisTiming(
        "omnivoice", "cuda", 32, 1000, backend_cls=OmniVoiceBackend,
        reference_of=lambda token: str(ref) if token == "Mara" else None)
    synthesize_chapter([Span(voice_id="Mara", text="Hello."), Span(voice_id=None, text="Hi.")],
                       lambda *a: torch.zeros(1, 1000), 1000, crossfade_ms=0, timing=timing)
    assert [r["ref_seconds"] for r in _rows(timing_db)] == [pytest.approx(12.0), None]
