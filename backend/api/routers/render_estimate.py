"""``POST /render/estimate``: how long a render will take where it will run.

docs/adr/render-time-estimate.md. Takes what a render takes (the Voice cloning
form for ``surface="generate"``, the ``/audiobook`` body for ``"audiobook"``,
the ``/longform/render`` body for ``"longform"``), plans the engine calls with
the same code the render runs (services/render_plan.py) and prices them with
the timings recorded for the target the render will actually go to: this
machine's calls (services/render_timing.py), or, when the GPU picker sends the
work to a remote worker, that worker's end-to-end task timings
(services/render_remote.py). No model load, no network, no write: safe to call
on every keystroke (debounced).
"""
from __future__ import annotations

import logging
import unicodedata
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import Field

from api.routers.audiobook import ExpressiveMixin, LongformChapter
from services import render_estimate as estimator
from services import render_plan, render_remote, render_timing, render_warmth

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


def _route(surface: str) -> render_remote.Route:
    """Where this surface's render would go right now (the GPU picker)."""
    return render_remote.route_for(_REMOTE_OPS[surface])


def _unavailable(engine_id: str, route: render_remote.Route) -> dict:
    return _with_target(estimator.unavailable_estimate(engine=render_timing.clean_id(engine_id)),
                        route)


def _with_target(result: dict, route: render_remote.Route) -> dict:
    """Name where the priced render runs (never the worker id)."""
    return {**result, "target": route.to_dict()}


def _pace(ref_audio: Optional[str], ref_text: Optional[str],
          ref_seconds: Optional[float] = None) -> render_plan.VoicePace:
    if ref_audio and ref_seconds is None:
        ref_seconds = render_warmth.reference_seconds(ref_audio)
    return render_plan.VoicePace(
        has_reference=bool(ref_audio) or ref_seconds is not None,
        ref_text=ref_text, ref_seconds=ref_seconds, path=ref_audio,
    )


def _warmups(backend_cls, planned: list[list[render_plan.PlannedCall]],
             pace_of) -> list[list[tuple[str, int]]]:
    """The cold costs the render will pay, per part, each with the position of
    the call that pays it: loading the engine before the first call when it is
    not loaded now, and ranking a long reference's passage before that voice's
    first call, or before every call on an engine that remembers no passage
    (services/render_warmth.py)."""
    warmups: list[list[tuple[str, int]]] = [[] for _ in planned]
    if planned and not render_warmth.engine_is_warm(backend_cls):
        warmups[0].append((render_warmth.LOAD, 0))
    every_call = render_warmth.ranks_every_call(backend_cls)
    seen: set = set()
    for index, part in enumerate(planned):
        for position, call in enumerate(part):
            pace = pace_of(call.voice)
            path = pace.path if pace is not None else None
            if path and (every_call or path not in seen):
                seen.add(path)
                if render_warmth.voice_needs_passage(backend_cls, path):
                    warmups[index].append((render_warmth.VOICE, position))
    return warmups


def _price(engine_id: str, backend_cls, num_step, planned: list[list[render_plan.PlannedCall]],
           pace_of, *, detail: bool = False) -> dict:
    engine = render_timing.engine_key(engine_id, backend_cls)
    omnivoice = render_plan.uses_omnivoice_estimator(backend_cls)
    rate = render_timing.seconds_per_char(engine)

    def ref_seconds(call: render_plan.PlannedCall) -> float:
        pace = pace_of(call.voice)
        return render_warmth.effective_reference_seconds(
            backend_cls, pace.ref_seconds if pace is not None else None)

    parts = [[(render_plan.call_audio_seconds(call, omnivoice=omnivoice, pace_of=pace_of,
                                              seconds_per_char=rate), ref_seconds(call))
              for call in part]
             for part in planned]
    return estimator.estimate(
        engine=engine,
        device=render_timing.device_class(backend_cls),
        num_step=render_timing.timing_steps(backend_cls, num_step),
        parts=parts,
        shape=render_warmth.call_shape(backend_cls),
        warmups=_warmups(backend_cls, planned, pace_of),
        detail=detail,
    )


def _price_remote(engine_id: str, backend_cls, num_step,
                  planned: list[list[render_plan.PlannedCall]], pace_of,
                  route: render_remote.Route, *, detail: bool = False) -> dict:
    """Price a render on the worker ``route`` names, from that worker's own
    bucket. A remote task is a whole unit (the take, or one chapter): each
    part is planned as one call carrying all of its audio, and the model
    load is added when the worker's heartbeats say the engine is not resident."""
    engine = render_remote.engine_for(engine_id)
    omnivoice = render_plan.uses_omnivoice_estimator(backend_cls)
    rate = render_timing.seconds_per_char(engine)
    parts: list[list[estimator.PlannedAudio]] = []
    for part in planned:
        audio = [render_plan.call_audio_seconds(call, omnivoice=omnivoice, pace_of=pace_of,
                                                seconds_per_char=rate) for call in part]
        known = all(a is not None for a in audio)
        parts.append([(sum(audio) if known else None, 0.0)] if part else [])
    warmups: list[list[str]] = [[] for _ in parts]
    first = next((index for index, part in enumerate(parts) if part), None)
    if first is not None and render_remote.worker_resident(route.worker_id, engine_id) is False:
        warmups[first].append(render_warmth.LOAD)  # paid by that part's first (only) task
    return estimator.estimate(
        engine=engine,
        device=route.device,
        num_step=render_timing.timing_steps(backend_cls, num_step),
        parts=parts,
        warmups=warmups,
        detail=detail,
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
    route = _route("generate")
    if route.kind == render_remote.UNAVAILABLE:
        return _unavailable(engine_id, route)

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
    if route.remote:
        return _with_target(_price_remote(engine_id, backend_cls, num_step, [calls],
                                          lambda _voice: pace, route), route)
    return _with_target(_price(engine_id, backend_cls, num_step, [calls], lambda _voice: pace),
                        route)


def _estimate_longform(req: RenderEstimateRequest) -> dict:
    from api.routers.audiobook import (
        _expressive_opts, _resolve_default_language, plan_from_chapters,
    )
    from services.audiobook import parse_audiobook_script
    from services.tts_backend import active_backend_id

    plan = (parse_audiobook_script(req.text or "", default_voice=req.default_voice)
            if req.surface == "audiobook" else plan_from_chapters(req.chapters or []))
    route = _route(req.surface)
    if route.kind == render_remote.UNAVAILABLE:
        return _unavailable(active_backend_id(), route)
    return plan_longform(
        plan, default_voice=req.default_voice, voice_map=req.voice_map,
        language=_resolve_default_language(req.language, req.default_voice),
        lexicon=req.lexicon, opts=_expressive_opts(req), route=route,
    )


def plan_longform(plan, *, default_voice, voice_map, language, lexicon, opts,
                  detail: bool = False,
                  route: render_remote.Route = render_remote.LOCAL_ROUTE) -> dict:
    """Plan and price a longform render (Audiobook, Stories) where it runs.

    ``plan`` is the parsed :class:`services.audiobook.AudiobookPlan` and
    ``language`` the already-resolved render language. Shared by the estimate
    endpoint and by a running render, which asks with ``detail`` for the
    per-call plan its live countdown follows (one call per chapter on a
    remote ``route``: a worker renders a chapter as one task).
    """
    from api.routers.audiobook import _engine_num_step, _map_span_voice, _resolve_voice
    from services.audiobook import normalized_spans
    from services.tts_backend import active_backend_id

    engine_id = active_backend_id()
    backend_cls = _backend_class(engine_id)
    profile_of: dict = {}
    paces: dict = {}

    def pace_of(token: Optional[str]) -> Optional[render_plan.VoicePace]:
        if token not in profile_of:
            profile_of[token] = _map_span_voice(token, default_voice, voice_map)
        profile_id = profile_of[token]
        if profile_id not in paces:
            voice = _resolve_voice(profile_id)
            paces[profile_id] = _pace(voice.get("ref_audio"), voice.get("ref_text"))
        return paces[profile_id]

    planned = [
        render_plan.chapter_calls(normalized_spans(chapter.spans, language),
                                  lexicon=lexicon,
                                  paragraph_gap_ms=opts.paragraph_gap_ms)
        for chapter in plan.chapters
    ]
    num_step = _engine_num_step(backend_cls, opts)
    if route.remote:
        return _with_target(_price_remote(engine_id, backend_cls, num_step, planned, pace_of,
                                          route, detail=detail), route)
    return _with_target(_price(engine_id, backend_cls, num_step, planned, pace_of,
                               detail=detail), route)


@router.post("/render/estimate")
def render_estimate(req: RenderEstimateRequest) -> dict:
    """Estimate a render's wall time from this machine's measured speed.

    Response: ``seconds``, ``low``, ``high`` (``None`` unless measured or
    rough), ``calls``, ``samples`` (warm calls the model is fitted on),
    ``basis`` (``measured`` | ``rough`` | ``none``), ``reason`` (why
    ``none``: ``cold_start``, ``remote_unavailable`` or ``no_rate``),
    ``target`` (``{"kind": "local"}``, or ``remote`` / ``unavailable`` with the
    worker's ``label``; ``offline`` for an unavailable one), ``audio_seconds``,
    ``warmup_seconds`` (measured cold-start cost included: an engine load, a
    long reference's first passage choice), the ``engine`` / ``device`` /
    ``num_step`` bucket, and ``parts``: calls and seconds per chapter (one
    part for ``generate``) for the live countdown.
    """
    if req.surface == "generate":
        return _estimate_generate(req)
    if req.surface == "longform" and req.chapters is None:
        raise HTTPException(status_code=422, detail="longform estimates need chapters")
    return _estimate_longform(req)
