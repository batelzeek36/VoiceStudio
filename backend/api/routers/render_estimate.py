"""``POST /render/estimate``: how long a render will take on THIS machine.

docs/adr/render-time-estimate.md. Takes what a render takes (the Voice cloning
form for ``surface="generate"``, the ``/audiobook`` body for ``"audiobook"``,
the ``/longform/render`` body for ``"longform"``), plans the engine calls with
the same code the render runs (services/render_plan.py) and prices them with
the timings this machine recorded (services/render_timing.py). No model load,
no network, no write: safe to call on every keystroke (debounced).
"""
from __future__ import annotations

import logging
import unicodedata
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import Field

from api.routers.audiobook import ExpressiveMixin, LongformChapter
from services import render_estimate as estimator
from services import render_plan, render_timing

router = APIRouter()
logger = logging.getLogger("omnivoice.render_estimate")

#: The remote-routing operation each surface's render asks about.
_REMOTE_OPS = {"generate": "tts", "audiobook": "audiobook", "longform": "longform"}


class RenderEstimateRequest(ExpressiveMixin):
    """What a render takes. Fields a surface does not use are ignored."""

    surface: Literal["generate", "audiobook", "longform"]
    # generate + audiobook
    text: Optional[str] = None
    language: Optional[str] = None
    # generate
    engine: Optional[str] = Field(default=None, max_length=64)
    profile_id: Optional[str] = Field(default=None, max_length=128)
    ref_text: Optional[str] = None
    ref_seconds: Optional[float] = Field(default=None, gt=0, le=3600)
    instruct: Optional[str] = None
    speed: float = Field(default=1.0, gt=0, le=4)
    duration: Optional[float] = Field(default=None, gt=0, le=3600)
    max_chunk_chars: int = Field(default=800, ge=0, le=1_000_000)
    pronounce: bool = True
    # audiobook + longform
    chapters: Optional[list[LongformChapter]] = None
    default_voice: Optional[str] = None
    voice_map: Optional[dict[str, str]] = None
    lexicon: Optional[dict] = None


def _backend_class(engine_id: str):
    from services.tts_backend import get_backend_class

    try:
        return get_backend_class(engine_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown TTS engine: {engine_id!r}") from None


def _runs_remotely(surface: str) -> bool:
    try:
        from services import gpu_gateway

        return bool(getattr(gpu_gateway.decide(_REMOTE_OPS[surface]), "remote", False))
    except Exception:  # noqa: BLE001 - routing is advisory; local always works
        return False


def _pace(ref_audio: Optional[str], ref_text: Optional[str],
          ref_seconds: Optional[float] = None) -> render_plan.VoicePace:
    if ref_audio and ref_seconds is None:
        from services.tts_backend import reference_duration_s

        ref_seconds = reference_duration_s(ref_audio)
    return render_plan.VoicePace(
        has_reference=bool(ref_audio) or ref_seconds is not None,
        ref_text=ref_text, ref_seconds=ref_seconds,
    )


def _price(engine_id: str, backend_cls, num_step, planned: list[list[render_plan.PlannedCall]],
           pace_of) -> dict:
    engine = render_timing.engine_key(engine_id, backend_cls)
    omnivoice = render_plan.uses_omnivoice_estimator(backend_cls)
    rate = render_timing.seconds_per_char(engine)
    parts = [[render_plan.call_audio_seconds(call, omnivoice=omnivoice, pace_of=pace_of,
                                             seconds_per_char=rate) for call in part]
             for part in planned]
    return estimator.estimate(
        engine=engine,
        device=render_timing.device_class(backend_cls),
        num_step=render_timing.timing_steps(backend_cls, num_step),
        parts=parts,
    )


def _estimate_generate(req: RenderEstimateRequest) -> dict:
    from services.performance_profiles import tts_defaults
    from services.synthesis_text import prepare_synthesis_text
    from services.tts_backend import active_backend_id

    # Same order as POST /generate: NFC, engine, steps, voice, language, text.
    text = unicodedata.normalize("NFC", req.text or "")
    engine_id = req.engine or active_backend_id()
    backend_cls = _backend_class(engine_id)
    num_step = req.num_step
    if num_step is None:
        num_step = tts_defaults(engine_id).get("num_step", 16)
    if _runs_remotely("generate"):
        return estimator.remote_estimate(engine=render_timing.clean_id(engine_id), calls=0)

    language = req.language
    pace = render_plan.VoicePace()
    if req.profile_id:
        from api.routers.generation import _resolve_profile_conditioning
        from core.db import db_conn

        with db_conn() as conn:
            row = conn.execute("SELECT * FROM voice_profiles WHERE id=?",
                               (req.profile_id,)).fetchone()
        if row:
            cond = _resolve_profile_conditioning(row, instruct=req.instruct or None,
                                                 language=language)
            language = cond["language"]
            pace = _pace(cond["ref_audio_path"], cond["ref_text"])
    elif req.ref_seconds is not None:
        pace = _pace(None, req.ref_text, req.ref_seconds)
    if isinstance(language, str) and language.strip().lower() == "auto":
        language = None
    text = prepare_synthesis_text(text, language, pronounce=req.pronounce)
    calls = render_plan.generate_calls(
        text, max_chunk_chars=req.max_chunk_chars, speed=req.speed, duration=req.duration,
    )
    return _price(engine_id, backend_cls, num_step, [calls], lambda _voice: pace)


def _estimate_longform(req: RenderEstimateRequest) -> dict:
    from api.routers.audiobook import (
        _engine_num_step, _expressive_opts, _map_span_voice, _resolve_default_language,
        _resolve_voice, plan_from_chapters,
    )
    from services.audiobook import normalized_spans, parse_audiobook_script
    from services.tts_backend import active_backend_id

    engine_id = active_backend_id()
    backend_cls = _backend_class(engine_id)
    opts = _expressive_opts(req)
    plan = (parse_audiobook_script(req.text or "", default_voice=req.default_voice)
            if req.surface == "audiobook" else plan_from_chapters(req.chapters or []))
    if _runs_remotely(req.surface):
        return estimator.remote_estimate(engine=render_timing.clean_id(engine_id),
                                         calls=0)

    language = _resolve_default_language(req.language, req.default_voice)
    profile_of: dict = {}
    paces: dict = {}

    def pace_of(token: Optional[str]) -> Optional[render_plan.VoicePace]:
        if token not in profile_of:
            profile_of[token] = _map_span_voice(token, req.default_voice, req.voice_map)
        profile_id = profile_of[token]
        if profile_id not in paces:
            voice = _resolve_voice(profile_id)
            paces[profile_id] = _pace(voice.get("ref_audio"), voice.get("ref_text"))
        return paces[profile_id]

    planned = [
        render_plan.chapter_calls(normalized_spans(chapter.spans, language),
                                  lexicon=req.lexicon,
                                  paragraph_gap_ms=opts.paragraph_gap_ms)
        for chapter in plan.chapters
    ]
    return _price(engine_id, backend_cls, _engine_num_step(backend_cls, opts), planned, pace_of)


@router.post("/render/estimate")
def render_estimate(req: RenderEstimateRequest) -> dict:
    """Estimate a render's wall time from this machine's measured speed.

    Response: ``seconds``, ``low``, ``high`` (``None`` unless measured or
    rough), ``calls``, ``samples``, ``basis`` (``measured`` | ``rough`` |
    ``none``), ``reason`` (why ``none``: ``cold_start``, ``remote`` or
    ``no_rate``), ``audio_seconds``, the ``engine`` / ``device`` /
    ``num_step`` bucket, and ``parts``: calls and seconds per chapter (one
    part for ``generate``) for the live countdown.
    """
    if req.surface == "generate":
        return _estimate_generate(req)
    if req.surface == "longform" and req.chapters is None:
        raise HTTPException(status_code=422, detail="longform estimates need chapters")
    return _estimate_longform(req)
