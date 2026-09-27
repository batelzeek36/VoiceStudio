"""Record what one synthesis call costs on THIS machine (render-time estimate).

docs/adr/render-time-estimate.md. Every completed synthesis call on the render
surfaces (generate, audiobook, longform, preview) adds one ``render_timings``
row: engine, resolved device class, unmasking steps, characters, audio seconds
produced, wall seconds and speed. Numbers and identifiers only, the same rule as
``core.render_trace``: no text, no voice, no path. The rows never leave the
machine (not analytics, not the diagnostic bundle) and are capped per bucket.

Each surface already routes its engine call through
``trace_call("synthesis", ...)``; it now goes through :meth:`SynthesisTiming.call`,
which wraps that same call and records the row. Recording can never break a
render: every failure here is logged at debug level and dropped.
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from core.render_trace import call as trace_call
from services.render_fit import FIT_WINDOW, Sample

logger = logging.getLogger("omnivoice.render_timing")

#: Newest rows kept per (engine, device, num_step) bucket (ADR).
MAX_ROWS_PER_BUCKET = 200
#: Backstop across every bucket, so a user sweeping the steps slider through
#: dozens of values cannot grow the table without bound.
MAX_ROWS_TOTAL = 5000
#: Rows the characters-per-second rate is learned from.
_RATE_WINDOW = 400
#: A lock held by another writer must not stall a render for long.
_BUSY_TIMEOUT_MS = 1000

_ID_RE = re.compile(r"[^a-z0-9._:-]+")

SampleRate = Union[int, float, Callable[[], Any]]


def clean_id(value: Any, *, limit: int = 64) -> str:
    """An identifier safe to store: lower-case ``[a-z0-9._:-]``, bounded."""
    return _ID_RE.sub("-", str(value or "").strip().lower()).strip("-")[:limit]


def honors_num_step(backend_cls: Any) -> bool:
    """Whether this engine's ``generate`` uses ``num_step`` at all."""
    return bool(getattr(backend_cls, "honors_num_step", False))


def timing_steps(backend_cls: Any, num_step: Any) -> Optional[int]:
    """The steps value a timing is bucketed under: ``None`` when the engine
    ignores steps (so a value it never reads cannot split its measurements) or
    when none was passed (the engine used its own default)."""
    if not honors_num_step(backend_cls) or num_step is None:
        return None
    try:
        steps = int(num_step)
    except (TypeError, ValueError):
        return None
    return steps if steps > 0 else None


def device_class(backend_cls: Any) -> str:
    """The device family this engine runs on here, without loading anything.

    Recorder and estimator both call this, so a render and its estimate always
    agree on the bucket: ``cuda``, ``rocm``, ``mps``, ``xpu``, ``npu``,
    ``directml``, a native runtime target (``vulkan``, ``metal``) or ``cpu``.
    """
    from core.device_caps import DIRECTML_MARKER, detect_host_caps
    from services.engine_routing import runtime_compute_profile

    caps = detect_host_caps()
    try:
        device = str(runtime_compute_profile(backend_cls, caps).get("effective_device") or "")
    except Exception:  # noqa: BLE001 - a routing probe failure must not break a render
        device = ""
    device = clean_id(device or caps.family, limit=16) or "cpu"
    if device == "cpu" and any(DIRECTML_MARKER in note for note in caps.notes):
        # The native OmniVoice loader (model_manager.get_best_device) is the
        # one path that runs on DirectML while the probe reports "cpu".
        from services.tts_backend import OmniVoiceBackend

        if backend_cls is OmniVoiceBackend:
            device = "directml"
    return device


def _overrides_model_identity(backend_cls: Any) -> bool:
    from services.tts_backend import TTSBackend

    method = getattr(backend_cls, "model_identity", None)
    return method is not None and method is not TTSBackend.model_identity


def engine_key(engine_id: str, backend_cls: Any = None) -> str:
    """The engine a timing belongs to: its id, plus the concrete model for an
    adapter that hosts several very different models behind one id
    (mlx-audio, cosyvoice, sherpa-onnx, audio.cpp).

    Derived from the class and the current configuration, never from a running
    instance, so the recorder and the estimator always name the same bucket.
    Never instantiates an out-of-process engine: that would register a sidecar
    owner per estimate."""
    ident = None
    try:
        if backend_cls is not None and _overrides_model_identity(backend_cls):
            if getattr(backend_cls, "runs_out_of_process", False):
                # Their identities read configuration, not instance state.
                ident = backend_cls.model_identity(object.__new__(backend_cls))
            else:
                ident = backend_cls().model_identity()
    except Exception:  # noqa: BLE001 - unknown model: fall back to the engine id
        ident = None
    base = clean_id(engine_id)
    model = clean_id(ident, limit=48) if ident else ""
    return f"{base}:{model}" if base and model else base


def _sample_count(audio: Any) -> int:
    if isinstance(audio, (list, tuple)):
        audio = audio[0] if audio else None
    shape = getattr(audio, "shape", None)
    try:
        if shape is not None and len(shape) > 0:
            return int(shape[-1])
        return len(audio) if audio is not None else 0
    except Exception:  # noqa: BLE001
        return 0


def _connect():
    from core import db as core_db

    conn = core_db.get_db()
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    return conn


def _insert(row: tuple) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO render_timings (engine, device, num_step, text_chars, "
            "audio_seconds, wall_seconds, speed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )
        engine, device, num_step = row[0], row[1], row[2]
        conn.execute(
            "DELETE FROM render_timings WHERE engine = ? AND device = ? AND num_step IS ? "
            "AND id NOT IN (SELECT id FROM render_timings WHERE engine = ? AND device = ? "
            "AND num_step IS ? ORDER BY id DESC LIMIT ?)",
            (engine, device, num_step, engine, device, num_step, MAX_ROWS_PER_BUCKET),
        )
        conn.execute(
            "DELETE FROM render_timings WHERE id NOT IN "
            "(SELECT id FROM render_timings ORDER BY id DESC LIMIT ?)",
            (MAX_ROWS_TOTAL,),
        )
        conn.commit()
    finally:
        conn.close()


def record(engine: str, device: str, num_step: Optional[int], text_chars: int,
           audio_seconds: float, wall_seconds: float, speed: Optional[float] = None) -> bool:
    """Store one completed call. Returns False (and stores nothing) for a call
    that produced no audio or carries a non-finite number. Never raises."""
    try:
        engine, device = clean_id(engine), clean_id(device, limit=16)
        values = (float(audio_seconds), float(wall_seconds))
        if not engine or not device or text_chars <= 0:
            return False
        if not all(math.isfinite(v) and v > 0 for v in values):
            return False
        spd = float(speed) if speed not in (None, 0) else None
        if spd is not None and not (math.isfinite(spd) and spd > 0):
            spd = None
        row = (engine, device, num_step, int(text_chars), values[0], values[1], spd, time.time())
        try:
            _insert(row)
        except Exception as exc:  # noqa: BLE001
            if "no such table" not in str(exc).lower():
                raise
            # A DB that missed init (#710 class): self-heal once, then retry.
            from core.db import ensure_schema

            ensure_schema()
            _insert(row)
        return True
    except Exception:  # noqa: BLE001 - a timing row is never worth a failed render
        logger.debug("render timing not recorded", exc_info=True)
        return False


@dataclass(frozen=True)
class SynthesisTiming:
    """Where one render's synthesis calls are measured and recorded."""

    engine: str
    device: str
    num_step: Optional[int]
    sample_rate: SampleRate

    def _rate(self) -> float:
        rate = self.sample_rate() if callable(self.sample_rate) else self.sample_rate
        return float(rate)

    def call(self, text: str, speed: Optional[float], fn: Callable, /, *args, **kwargs):
        """``trace_call("synthesis", fn, ...)``, recording the call once it succeeds.

        ``text`` is only counted (its length), never stored. The sample rate is
        read after the call because lazily loading engines report their real
        rate only once their weights are up.
        """
        start = time.perf_counter()
        out = trace_call("synthesis", fn, *args, **kwargs)
        wall = time.perf_counter() - start
        try:
            samples = _sample_count(out)
            rate = self._rate()
            if samples > 0 and rate > 0:
                record(self.engine, self.device, self.num_step, len(text or ""),
                       samples / rate, wall, speed)
        except Exception:  # noqa: BLE001
            logger.debug("render timing not recorded", exc_info=True)
        return out


class _Untimed:
    """The same call with nothing recorded (paths outside the render surfaces)."""

    def call(self, text: str, speed: Optional[float], fn: Callable, /, *args, **kwargs):
        return trace_call("synthesis", fn, *args, **kwargs)


UNTIMED = _Untimed()
Timing = Union[SynthesisTiming, _Untimed]


def for_render(engine_id: str, backend_cls: Any, *, num_step: Any,
               sample_rate: SampleRate) -> Timing:
    """The timing context for one local render; :data:`UNTIMED` if it cannot
    be resolved (never raises, so it can sit in front of any render)."""
    try:
        engine = engine_key(engine_id, backend_cls)
        if not engine:
            return UNTIMED
        return SynthesisTiming(
            engine=engine,
            device=device_class(backend_cls),
            num_step=timing_steps(backend_cls, num_step),
            sample_rate=sample_rate,
        )
    except Exception:  # noqa: BLE001
        logger.debug("render timing context unavailable", exc_info=True)
        return UNTIMED


# ── Reads (the estimator) ───────────────────────────────────────────────────


def samples_for(engine: str, device: str, num_step: Optional[int]) -> list[Sample]:
    """Newest samples for this engine and device, newest first: the whole
    newest :data:`FIT_WINDOW` plus this steps bucket's own newest window."""
    conn = _connect()
    try:
        cols = "SELECT id, audio_seconds, wall_seconds, num_step FROM render_timings "
        rows = conn.execute(
            cols + "WHERE engine = ? AND device = ? ORDER BY id DESC LIMIT ?",
            (engine, device, FIT_WINDOW),
        ).fetchall()
        rows += conn.execute(
            cols + "WHERE engine = ? AND device = ? AND num_step IS ? ORDER BY id DESC LIMIT ?",
            (engine, device, num_step, FIT_WINDOW),
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        if "no such table" in str(exc).lower():
            return []
        raise
    finally:
        conn.close()
    unique = {row[0]: row for row in rows}
    return [Sample(audio_seconds=r[1], wall_seconds=r[2], num_step=r[3])
            for _, r in sorted(unique.items(), key=lambda item: item[0], reverse=True)]


def seconds_per_char(engine: str) -> Optional[float]:
    """Audio seconds one character yields at speed 1 for this engine, learned
    from this machine's own renders (any device: speech rate is not a device
    property). ``None`` before the first render."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT SUM(audio_seconds * COALESCE(speed, 1.0)), SUM(text_chars) FROM "
            "(SELECT audio_seconds, speed, text_chars FROM render_timings "
            "WHERE engine = ? ORDER BY id DESC LIMIT ?)",
            (engine, _RATE_WINDOW),
        ).fetchone()
    except Exception as exc:  # noqa: BLE001
        if "no such table" in str(exc).lower():
            return None
        raise
    finally:
        conn.close()
    if not row or not row[0] or not row[1]:
        return None
    return float(row[0]) / float(row[1])


__all__ = [
    "MAX_ROWS_PER_BUCKET", "MAX_ROWS_TOTAL", "SynthesisTiming", "Timing", "UNTIMED",
    "clean_id", "device_class", "engine_key", "for_render", "honors_num_step", "record",
    "samples_for", "seconds_per_char", "timing_steps",
]
