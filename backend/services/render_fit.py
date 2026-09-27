"""Robust per-call render-time model (docs/adr/render-time-estimate.md).

One synthesis call on this machine costs ``wall = a + b * audio_seconds``:
``a`` is the fixed cost every call pays (text encoding, reference prompt,
vocoder start-up, process hop) and ``b`` is the compute per second of audio.
Both are fitted from the machine's own ``render_timings`` rows, never assumed:

* Theil-Sen slope (the median of every pairwise slope) and a median intercept,
  so a warm-up call, a thermal stall or a render that raced another job moves
  nothing. Clamped to ``a >= 0`` and ``b > 0``.
* The spread of the samples around the line (25th to 75th percentile of
  observed / predicted) is the range.
* Fewer than :data:`MIN_SAMPLES` samples: no fit at all. Any prior would be
  right on one machine and wrong on the next.

Pure math on plain floats: no database, no torch, the same answer on every OS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Optional, Sequence

#: The estimate stays silent below this many samples for an engine and device.
MIN_SAMPLES = 3
#: Newest samples a fit reads. Old enough rows describe an older driver, torch
#: or thermal state; 120 keeps the pairwise slope set small (7,140 pairs).
FIT_WINDOW = 120
#: Two calls whose audio lengths differ by less than this carry no slope
#: information: their difference is timing noise divided by almost nothing.
_MIN_DX = 0.25
#: Extra spread for a steps-scaled ("rough") fit: compute is proportional to
#: unmasking steps only approximately (text encoding and decoding are not).
_ROUGH_LOW = 0.8
_ROUGH_HIGH = 1.25


@dataclass(frozen=True)
class Sample:
    """One completed synthesis call."""

    audio_seconds: float
    wall_seconds: float
    num_step: Optional[int] = None


@dataclass(frozen=True)
class Fit:
    """``wall = intercept + slope * audio_seconds`` for one call."""

    intercept: float
    slope: float
    low_ratio: float
    high_ratio: float
    samples: int
    rough: bool = False

    def predict(self, audio_seconds: float) -> float:
        return self.intercept + self.slope * max(0.0, float(audio_seconds))


def _finite_positive(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile (numpy's default), 0 <= pct <= 100."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * max(0.0, min(100.0, pct)) / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def fit_line(points: Iterable[tuple[float, float]], *, rough: bool = False) -> Optional[Fit]:
    """Fit ``wall = a + b * x`` robustly; ``None`` below :data:`MIN_SAMPLES`."""
    pts = [(float(x), float(y)) for x, y in points
           if _finite_positive(x) and _finite_positive(y)]
    if len(pts) < MIN_SAMPLES:
        return None
    slopes = [
        (y2 - y1) / (x2 - x1)
        for i, (x1, y1) in enumerate(pts)
        for x2, y2 in pts[i + 1:]
        if abs(x2 - x1) >= _MIN_DX
    ]
    slope = median(slopes) if slopes else 0.0
    intercept = median(y - slope * x for x, y in pts) if slope > 0 else -1.0
    if slope <= 0 or intercept < 0:
        # No usable spread in audio length, a negative slope (noise swamping
        # the signal) or a line that would need a negative per-call cost:
        # wall time proportional to the audio is the honest remaining model.
        slope = median(y / x for x, y in pts)
        intercept = 0.0
    ratios = [y / (intercept + slope * x) for x, y in pts]
    low = min(1.0, percentile(ratios, 25))
    high = max(1.0, percentile(ratios, 75))
    if rough:
        low *= _ROUGH_LOW
        high *= _ROUGH_HIGH
    return Fit(intercept=intercept, slope=slope, low_ratio=low, high_ratio=high,
               samples=len(pts), rough=rough)


def _step_ratio(sample_step: Optional[int], target_step: Optional[int]) -> float:
    if not sample_step or not target_step:
        return 1.0
    return float(sample_step) / float(target_step)


def fit_bucket(samples: Sequence[Sample], num_step: Optional[int]) -> Optional[Fit]:
    """The model for calls at ``num_step``, from one engine+device's samples.

    ``samples`` is newest first. Measured when this exact steps bucket has
    :data:`MIN_SAMPLES` samples. Otherwise, when the engine and device have
    that many at any steps, every sample is rescaled to ``num_step`` and the
    fit is marked rough: compute scales with unmasking steps, so a sample at
    ``s`` steps producing ``x`` seconds costs what ``x * s / num_step`` seconds
    would cost at ``num_step`` — which scales ``b`` by the steps ratio and
    leaves the per-call cost ``a`` alone. ``None`` below that: cold start.
    """
    exact = [s for s in samples if s.num_step == num_step][:FIT_WINDOW]
    measured = fit_line((s.audio_seconds, s.wall_seconds) for s in exact)
    if measured is not None:
        return measured
    pooled = list(samples)[:FIT_WINDOW]
    return fit_line(
        ((s.audio_seconds * _step_ratio(s.num_step, num_step), s.wall_seconds) for s in pooled),
        rough=True,
    )


__all__ = ["FIT_WINDOW", "Fit", "MIN_SAMPLES", "Sample", "fit_bucket", "fit_line", "percentile"]
