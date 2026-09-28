"""Which synthesis calls are cold, read from the engine's real state
(docs/adr/render-time-estimate.md).

A cold call costs more than the same call warm, for a reason that is not the
call itself. Two kinds are detected, each from a signal the app already keeps:

* ``load``: the first call on a freshly loaded engine. The signal is the object
  that a (re)load replaces: a subprocess engine's sidecar process (OmniVoice on
  MPS runs in one and loads its model on the first synthesize), the native
  OmniVoice model object held by ``model_manager``, or an in-process adapter's
  lazily loaded model attribute. A call on an object that has never finished a
  call here is cold; a sidecar that died or was reaped for idleness comes back
  as a new process, so its first call is cold again.
* ``voice``: the first call on a reference longer than the engine's
  ``max_ref_seconds`` whose 15 s passage has not been chosen yet. Choosing it
  ranks the windows with the installed recognizer inside the call; the choice is
  then remembered (``tts_backend._recall_passage``).

What is NOT a per-voice cold cost, from reading the code and measuring: the
OmniVoice sidecar has no prompt cache and encodes the reference on every call,
so a longer reference costs more on EVERY call; that is modelled as sequence
length (services/render_fit.py), not as a cold start. The native path's prompt
cache saves about 0.4 s per new voice, below timing noise, so it is not flagged.
An adapter that reloads weights without replacing an attribute we can see is
not detected; its first call after such a reload is recorded as warm.
"""
from __future__ import annotations

import logging
import threading
import weakref
from typing import Any, Optional

from services.render_fit import OMNIVOICE, PLAIN, CallShape

logger = logging.getLogger("omnivoice.render_warmth")

LOAD = "load"
VOICE = "voice"

#: Attributes adapters keep their lazily loaded model in; a reload replaces it.
_MODEL_ATTRS = ("_model", "_tts", "_session", "_pipeline")
#: Bound on the id fallback for tokens that cannot be weakly referenced.
_MAX_ID_TOKENS = 64

_lock = threading.Lock()
_warm_refs: "weakref.WeakSet[Any]" = weakref.WeakSet()
_warm_ids: list[int] = []


def omnivoice_family(backend_cls: Any) -> bool:
    """The engines that run the OmniVoice model itself (in process, or in the
    MPS or crash-isolated sidecar)."""
    from services.render_plan import uses_omnivoice_estimator

    return uses_omnivoice_estimator(backend_cls)


def call_shape(backend_cls: Any) -> CallShape:
    """How this engine turns one call into model passes (see render_fit)."""
    return OMNIVOICE if omnivoice_family(backend_cls) else PLAIN


def effective_reference_seconds(backend_cls: Any, raw_seconds: Optional[float]) -> float:
    """The reference audio the engine actually conditions on: a reference over
    ``max_ref_seconds`` is cut to its best 15 s window (OmniVoice) or capped."""
    if not raw_seconds or raw_seconds <= 0:
        return 0.0
    cap = getattr(backend_cls, "max_ref_seconds", None)
    if cap and raw_seconds > cap:
        if getattr(backend_cls, "ref_strategy", None) == "best_window":
            from omnivoice.utils.audio import CLONE_REF_WINDOW_SECONDS

            return float(CLONE_REF_WINDOW_SECONDS)
        return float(cap)
    return float(raw_seconds)


def reference_seconds(path: Optional[str]) -> Optional[float]:
    """The raw length of a reference file (cached per file version)."""
    if not path:
        return None
    from services.tts_backend import reference_duration_s

    return reference_duration_s(path)


def runtime_token(runtime: Any) -> Any:
    """The object a (re)load of ``runtime`` replaces; ``None`` when nothing is
    loaded right now (the next call will load it)."""
    if runtime is None:
        return None
    state = getattr(runtime, "__dict__", {})
    if "_proc" in state:
        proc = state["_proc"]
        if proc is None:
            return None
        poll = getattr(proc, "poll", None)
        try:
            if callable(poll) and poll() is not None:
                return None  # the sidecar exited; the next call respawns it
        except Exception:  # noqa: BLE001
            return None
        return proc
    from services.tts_backend import TTSBackend

    if isinstance(runtime, TTSBackend):
        for attr in _MODEL_ATTRS:
            if attr in state:
                return state[attr]
    return runtime


def _is_warm_token(token: Any) -> bool:
    if token is None:
        return False
    with _lock:
        try:
            if token in _warm_refs:
                return True
        except TypeError:
            pass
        return id(token) in _warm_ids


def mark_warm(runtime: Any) -> None:
    """``runtime`` just finished a call: later calls on it are warm."""
    token = runtime_token(runtime)
    if token is None:
        return
    with _lock:
        try:
            _warm_refs.add(token)
            return
        except TypeError:
            pass
        if id(token) not in _warm_ids:
            _warm_ids.append(id(token))
            del _warm_ids[:-_MAX_ID_TOKENS]


def is_warm(runtime: Any) -> bool:
    return _is_warm_token(runtime_token(runtime))


def live_runtime(backend_cls: Any) -> Any:
    """What a render on ``backend_cls`` would run on right now, without loading
    or creating anything: the native model ``model_manager`` holds, else the
    process-wide engine instance /generate and the audiobook share."""
    from services.tts_backend import OmniVoiceBackend

    if backend_cls is OmniVoiceBackend:
        from services import model_manager

        return getattr(model_manager, "model", None)
    from services import tts_backend

    return tts_backend._ENGINE_INSTANCES.get(backend_cls)


def engine_is_warm(backend_cls: Any) -> bool:
    try:
        return is_warm(live_runtime(backend_cls))
    except Exception:  # noqa: BLE001 - unknown state: assume the load is paid
        return False


def voice_needs_passage(backend_cls: Any, ref_path: Optional[str]) -> bool:
    """True when the next call on this reference will first rank its passage."""
    if not ref_path or not omnivoice_family(backend_cls):
        return False
    cap = getattr(backend_cls, "max_ref_seconds", None)
    raw = reference_seconds(ref_path)
    if not cap or raw is None or raw <= cap:
        return False
    try:
        from services.tts_backend import _recall_passage

        return _recall_passage(ref_path) is None
    except Exception:  # noqa: BLE001
        return False


def cold_kind(backend_cls: Any, runtime: Any, ref_path: Optional[str]) -> Optional[str]:
    """``load``, ``voice`` or ``None`` for a call about to run. Never raises."""
    try:
        if runtime is not None and not is_warm(runtime):
            return LOAD
        if voice_needs_passage(backend_cls, ref_path):
            return VOICE
    except Exception:  # noqa: BLE001
        logger.debug("cold-call signal unavailable", exc_info=True)
    return None


def _reset_for_tests() -> None:
    with _lock:
        _warm_refs.clear()
        _warm_ids.clear()


__all__ = [
    "LOAD", "VOICE", "call_shape", "cold_kind", "effective_reference_seconds",
    "engine_is_warm", "is_warm", "live_runtime", "mark_warm", "omnivoice_family",
    "reference_seconds", "runtime_token", "voice_needs_passage",
]
