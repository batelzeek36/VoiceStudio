"""Every audiobook chapter reuses the one process-wide engine instance.

``_build_synth`` runs once per chapter. It used to construct a new adapter each
time (``cls()``), so every chapter re-loaded the engine's weights; for a
subprocess engine (OmniVoice on MPS runs as one) every chapter spawned a new
sidecar and model while the earlier chapters' sidecars stayed resident until
the idle reaper. /generate already shares one cached instance per engine.
"""
import pytest
import torch


@pytest.fixture
def counting_engine(monkeypatch):
    import api.routers.audiobook as ab
    from services import tts_backend

    made = []

    class CountingEngine:
        sample_rate = 24000

        def __init__(self):
            made.append(self)

        def generate(self, text, **kwargs):
            return torch.zeros(1, 240)

    monkeypatch.setattr(tts_backend, "_ENGINE_INSTANCES", {})
    monkeypatch.setattr(tts_backend, "_ENGINE_LAST_USED", {})
    monkeypatch.setattr(tts_backend, "active_backend_id", lambda: "counting")
    monkeypatch.setattr(tts_backend, "get_backend_class", lambda engine_id: CountingEngine)
    monkeypatch.setattr(ab, "_resolve_voice", lambda _pid: {
        "ref_audio": None, "ref_text": None, "instruct": None, "seed": None,
    })
    return ab, made, CountingEngine


def test_chapters_share_one_engine_instance(counting_engine):
    ab, made, cls = counting_engine
    for _chapter in range(3):
        info = ab._build_synth("voice")
        info["synth"]("A line of the chapter.", None)
    assert len(made) == 1


def test_the_instance_is_the_one_generate_uses(counting_engine):
    from services.tts_backend import get_engine_instance

    ab, made, cls = counting_engine
    ab._build_synth("voice")["synth"]("Hello.", None)
    assert get_engine_instance(cls) is made[0]
    assert len(made) == 1
