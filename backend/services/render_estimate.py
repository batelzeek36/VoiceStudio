"""Turn a planned render into a time estimate from this machine's own timings.

docs/adr/render-time-estimate.md, parts 2 and 3. ``parts`` are the planned
calls, one list per progress unit the UI can see finish (one per chapter for
longform, one for ``/generate``); each call is ``(audio_seconds,
ref_seconds)``. ``warmups`` lists, per part, the cold costs the render will pay
when it reaches that part: ``load`` when the engine is not loaded right now,
``voice`` once for each long reference whose passage is not chosen yet.

Basis:

* ``measured``: this engine, device and steps have at least three warm calls,
  every planned pass lies within the measured lengths, and any cold cost the
  render will pay has been measured here.
* ``rough``: scaled from other steps, a planned pass longer or shorter than
  anything measured, or a cold cost not measured yet; the range is wider.
* ``none``: fewer than three warm calls here, or the render will not run on
  this machine. No number is better than a number that is wrong somewhere.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

from services import render_fit, render_timing

#: Why there is no number (``basis == "none"``).
REASON_COLD_START = "cold_start"
REASON_REMOTE = "remote"
REASON_NO_RATE = "no_rate"

PlannedAudio = tuple[Optional[float], float]


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(max(0.0, float(value)), 1)


def _none(base: dict, parts: Sequence[Sequence[PlannedAudio]], reason: str) -> dict:
    return {
        **base, "basis": "none", "reason": reason,
        "seconds": None, "low": None, "high": None, "warmup_seconds": None,
        "parts": [{"calls": len(part), "seconds": None} for part in parts],
    }


def remote_estimate(*, engine: str, calls: int) -> dict:
    """The render goes to a remote worker: this machine's speed says nothing."""
    base = {"engine": engine, "device": None, "num_step": None, "calls": calls,
            "audio_seconds": None, "samples": 0}
    return _none(base, [], REASON_REMOTE)


def estimate(*, engine: str, device: str, num_step: Optional[int],
             parts: Sequence[Sequence[PlannedAudio]],
             shape: render_fit.CallShape = render_fit.PLAIN,
             warmups: Optional[Sequence[Sequence[str]]] = None,
             detail: bool = False) -> dict:
    """Price ``parts`` with the model for ``(engine, device, num_step)``.

    ``detail`` adds each part's ``call_seconds`` (its warm-up folded into its
    first call), the plan a running render's live countdown follows."""
    samples = render_timing.samples_for(engine, device, num_step)
    model = render_fit.fit_model(samples, num_step, shape, now=time.time())
    warmups = list(warmups or [[] for _ in parts])
    calls = sum(len(part) for part in parts)
    known = all(audio is not None for part in parts for audio, _ in part)
    audio = sum(a for part in parts for a, _ in part if a is not None) if known else None
    base = {
        "engine": engine, "device": device, "num_step": num_step, "calls": calls,
        "audio_seconds": _round(audio),
        "samples": model.samples if model is not None
        else sum(1 for s in samples if not s.cold),
    }
    if model is None:
        return _none(base, parts, REASON_COLD_START)
    if not known:
        return _none(base, parts, REASON_NO_RATE)
    rough = model.rough
    totals = [0.0, 0.0, 0.0]
    warmup = 0.0
    out_parts = []
    for index, part in enumerate(parts):
        seconds = low = high = 0.0
        first: Optional[float] = None
        per_call: list[float] = []
        for audio_seconds, ref_seconds in part:
            p = model.predict(audio_seconds, ref_seconds)
            first = p.seconds if first is None else first
            per_call.append(p.seconds)
            seconds, low, high = seconds + p.seconds, low + p.low, high + p.high
            rough = rough or p.extrapolated
        for kind in warmups[index] if index < len(warmups) else []:
            overhead = model.overheads.get(kind)
            if overhead is not None:
                seconds, low, high = (seconds + overhead.typical, low + overhead.low,
                                      high + overhead.high)
                warmup += overhead.typical
                if per_call:
                    per_call[0] += overhead.typical
            else:
                # Signal read, cost never measured here: no number to add, so
                # allow up to one more first call and say it is rough.
                rough = True
                high += first or 0.0
        entry = {"calls": len(part), "seconds": _round(seconds)}
        if detail:
            entry["call_seconds"] = [round(c, 2) for c in per_call]
        out_parts.append(entry)
        totals = [totals[0] + seconds, totals[1] + low, totals[2] + high]
    return {
        **base,
        "basis": "rough" if rough else "measured",
        "reason": None,
        "seconds": _round(totals[0]),
        "low": _round(totals[1]),
        "high": _round(totals[2]),
        "warmup_seconds": _round(warmup),
        "pace": round(model.pace, 2),
        "parts": out_parts,
    }


__all__ = ["REASON_COLD_START", "REASON_NO_RATE", "REASON_REMOTE", "estimate", "remote_estimate"]
