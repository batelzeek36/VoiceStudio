"""Plan a render's engine calls without rendering (render-time estimate).

docs/adr/render-time-estimate.md, part 3. Every ``[pause]``, ``[voice:]``
switch, inline-markup boundary and chunk is a separate engine call with its own
fixed cost, so the estimate has to know the calls a render will make, not just
how much text there is. These planners call the same helpers the renders call:

* ``/generate``: ``parse_pause_markers`` then, without pauses, the sentence
  chunker, exactly the branch ``_run_inference`` / ``_run_backend_inference``
  take (a single call keeps the request's explicit ``duration``).
* longform (Audiobook, Stories, chapter previews): the spans the longform
  parser produced, normalized by ``normalized_spans``, respelled and split into
  paragraphs by ``span_paragraphs``, then chunked, exactly as
  ``synthesize_chapter`` renders them.

Audio seconds per call come from the engine's own duration estimate where it
has one (OmniVoice's rule estimator, fed the voice's reference transcript and
length, the way ``OmniVoice._estimate_target_tokens`` sizes its output), else
from characters per second learned from this machine's renders of that engine.
Nothing here loads a model or touches the network.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

#: OmniVoice's audio tokenizer: 24 kHz with a 960-sample hop (its
#: audio_tokenizer/config.json), i.e. 25 tokens per second of audio.
OMNIVOICE_FRAME_RATE = 25
#: ``OmniVoice._estimate_target_tokens`` sizes a voice with no reference
#: transcript (voice design, the engine default) from this pair.
_FALLBACK_REF_TEXT = "Nice to meet you."
_FALLBACK_REF_TOKENS = 25


@dataclass(frozen=True)
class PlannedCall:
    """One engine call a render will make."""

    text: str
    speed: float = 1.0
    voice: Optional[str] = None
    #: Explicit target length (``/generate``'s ``duration``, single call only).
    duration: Optional[float] = None


@dataclass(frozen=True)
class VoicePace:
    """What a voice's reference says about how fast it speaks."""

    has_reference: bool = False
    ref_text: Optional[str] = None
    ref_seconds: Optional[float] = None
    #: The reference file, when it is one on disk (never sent anywhere).
    path: Optional[str] = None

    @property
    def known(self) -> bool:
        return bool(self.ref_text and self.ref_text.strip() and self.ref_seconds
                    and self.ref_seconds > 0)


def _speed(value) -> float:
    try:
        speed = float(value) if value else 1.0
    except (TypeError, ValueError):
        return 1.0
    return speed if speed > 0 else 1.0


def generate_calls(text: str, *, max_chunk_chars: Optional[int], speed=1.0,
                   duration: Optional[float] = None,
                   voice: Optional[str] = None) -> list[PlannedCall]:
    """The engine calls ``POST /generate`` makes for already-prepared text."""
    from omnivoice.utils.text import parse_pause_markers
    from services.chunked_tts import DEFAULT_MAX_CHUNK_CHARS, split_text_into_chunks

    if not text or not text.strip():
        return []
    spd = _speed(speed)
    segments = parse_pause_markers(text)
    if len(segments) > 1 or (segments and segments[0][1] > 0):
        # _render_with_pauses: one call per span with speakable text; the
        # overall duration cannot be split across spans, so it is dropped.
        return [PlannedCall(span, spd, voice) for span, _ in segments if span and span.strip()]
    max_chars = DEFAULT_MAX_CHUNK_CHARS if max_chunk_chars is None else max_chunk_chars
    chunks = split_text_into_chunks(text, max_chars)
    if len(chunks) > 1:
        return [PlannedCall(chunk, spd, voice) for chunk in chunks]
    return [PlannedCall(text, spd, voice, duration)]


def chapter_calls(spans: Iterable, *, lexicon: Optional[dict],
                  paragraph_gap_ms: int) -> list[PlannedCall]:
    """The engine calls ``synthesize_chapter`` makes for normalized spans.

    ``voice`` on each call is the span's raw voice token; the caller maps it to
    a profile the way the render does.
    """
    from services.audiobook import span_paragraphs
    from services.chunked_tts import split_text_into_chunks

    calls: list[PlannedCall] = []
    for span in spans:
        spd = _speed(getattr(span, "speed", None))
        for paragraph in span_paragraphs(span.text, lexicon, paragraph_gap_ms):
            calls.extend(PlannedCall(chunk, spd, span.voice_id)
                         for chunk in split_text_into_chunks(paragraph))
    return calls


@functools.lru_cache(maxsize=1)
def _estimator():
    from omnivoice.utils.duration import RuleDurationEstimator

    return RuleDurationEstimator()


def omnivoice_seconds(text: str, pace: Optional[VoicePace], speed: float = 1.0) -> float:
    """Audio seconds OmniVoice will generate for ``text`` in this voice.

    Mirrors ``OmniVoice._estimate_target_tokens``: the rule estimator scales
    the reference's tokens-per-weight to the target text (with its short-text
    boost), divides by speed and floors to whole tokens.
    """
    if pace is not None and pace.known:
        ref_text = pace.ref_text
        ref_tokens = max(1, int(round(pace.ref_seconds * OMNIVOICE_FRAME_RATE)))
    else:
        ref_text, ref_tokens = _FALLBACK_REF_TEXT, _FALLBACK_REF_TOKENS
    est = _estimator().estimate_duration(text, ref_text, ref_tokens)
    spd = _speed(speed)
    if spd != 1.0:
        est = est / spd
    return max(1, int(est)) / OMNIVOICE_FRAME_RATE


def uses_omnivoice_estimator(backend_cls) -> bool:
    """True for the engines that run the OmniVoice model itself (in process,
    in the MPS or crash-isolated sidecar): their output length IS the rule
    estimator's target length."""
    from services.tts_backend import OmniVoiceBackend

    if backend_cls is None:
        return False
    if isinstance(backend_cls, type) and issubclass(backend_cls, OmniVoiceBackend):
        return True
    try:
        from engines.omnivoice_subprocess import OmniVoiceSubprocessBackend
    except Exception:  # noqa: BLE001 - the sidecar package is optional
        return False
    return isinstance(backend_cls, type) and issubclass(backend_cls, OmniVoiceSubprocessBackend)


def call_audio_seconds(
    call: PlannedCall,
    *,
    omnivoice: bool,
    pace_of: Callable[[Optional[str]], Optional[VoicePace]],
    seconds_per_char: Optional[float],
) -> Optional[float]:
    """Audio seconds one planned call produces; ``None`` when nothing on this
    machine says yet how fast this engine speaks."""
    if call.duration is not None and call.duration > 0:
        return float(call.duration)
    if omnivoice:
        pace = pace_of(call.voice)
        if pace is None or pace.known or not pace.has_reference or not seconds_per_char:
            # A reference with no transcript is transcribed at render time; until
            # this machine has measured the engine, OmniVoice's own default rate
            # is the model's answer too.
            return omnivoice_seconds(call.text, pace, call.speed)
    if seconds_per_char:
        return len(call.text) * seconds_per_char / _speed(call.speed)
    return None


__all__ = [
    "OMNIVOICE_FRAME_RATE", "PlannedCall", "VoicePace", "call_audio_seconds",
    "chapter_calls", "generate_calls", "omnivoice_seconds", "uses_omnivoice_estimator",
]
