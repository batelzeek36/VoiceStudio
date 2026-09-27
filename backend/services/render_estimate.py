"""Turn a planned render into a time estimate from this machine's own timings.

docs/adr/render-time-estimate.md, parts 2 and 3. ``parts`` are the planned
calls' audio seconds, one list per progress unit the UI can see finish (one
per chapter for longform, one for ``/generate``), so a live countdown can
re-fit as each unit completes.

Basis:

* ``measured``: this engine, device and steps have at least three samples.
* ``rough``: this engine and device have them at other steps; the fit is
  rescaled by the steps ratio (see :func:`services.render_fit.fit_bucket`).
* ``none``: fewer than three samples here, or the render will not run on this
  machine. No number is better than a number that is wrong somewhere.
"""
from __future__ import annotations

from typing import Optional, Sequence

from services import render_fit, render_timing

#: Why there is no number (``basis == "none"``).
REASON_COLD_START = "cold_start"
REASON_REMOTE = "remote"
REASON_NO_RATE = "no_rate"


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(max(0.0, float(value)), 1)


def _none(base: dict, parts: Sequence[Sequence[Optional[float]]], reason: str) -> dict:
    return {
        **base, "basis": "none", "reason": reason,
        "seconds": None, "low": None, "high": None,
        "parts": [{"calls": len(part), "seconds": None} for part in parts],
    }


def remote_estimate(*, engine: str, calls: int) -> dict:
    """The render goes to a remote worker: this machine's speed says nothing."""
    base = {"engine": engine, "device": None, "num_step": None, "calls": calls,
            "audio_seconds": None, "samples": 0}
    return _none(base, [], REASON_REMOTE)


def estimate(*, engine: str, device: str, num_step: Optional[int],
             parts: Sequence[Sequence[Optional[float]]]) -> dict:
    """Price ``parts`` with the fit for ``(engine, device, num_step)``."""
    samples = render_timing.samples_for(engine, device, num_step)
    fit = render_fit.fit_bucket(samples, num_step)
    calls = sum(len(part) for part in parts)
    known = all(x is not None for part in parts for x in part)
    audio = sum(x for part in parts for x in part if x is not None) if known else None
    base = {
        "engine": engine, "device": device, "num_step": num_step, "calls": calls,
        "audio_seconds": _round(audio),
        "samples": fit.samples if fit is not None else len(samples),
    }
    if fit is None:
        return _none(base, parts, REASON_COLD_START)
    if not known:
        return _none(base, parts, REASON_NO_RATE)
    part_seconds = [sum(fit.predict(x) for x in part) for part in parts]
    total = sum(part_seconds)
    return {
        **base,
        "basis": "rough" if fit.rough else "measured",
        "reason": None,
        "seconds": _round(total),
        "low": _round(total * fit.low_ratio),
        "high": _round(total * fit.high_ratio),
        "parts": [{"calls": len(part), "seconds": _round(sec)}
                  for part, sec in zip(parts, part_seconds)],
    }


__all__ = ["REASON_COLD_START", "REASON_NO_RATE", "REASON_REMOTE", "estimate", "remote_estimate"]
