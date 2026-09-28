"""The render-time model, fitted from this machine's own calls
(docs/adr/render-time-estimate.md).

What one synthesis call costs is driven by the sequence the model attends over,
not by the call's audio length alone. Measured on an M1 Max (OmniVoice, MPS, 64
steps), a single line fitted on audio seconds missed by a median 26%, while
this model misses by 3%:

* **Passes.** OmniVoice generates a call whose estimated audio exceeds 30 s as
  passes of about 15 s (``audio_chunk_threshold`` / ``audio_chunk_duration``
  in omnivoice/models/omnivoice.py), each paying the fixed cost again.
* **Reference.** Every OmniVoice pass carries the voice's reference audio in
  its sequence, so a 15 s reference costs more per call than an 11 s one.

So each call is split into passes (:class:`CallShape`), and one robust line is
fitted per pass: ``wall = a + b * length``, where ``length`` is the pass's
audio plus, for OmniVoice, its reference. Theil-Sen slope, median intercept,
``a >= 0``, ``b > 0``; the range is the spread of cross-validated errors
(10th to 90th percentile, never tighter than 10% either way). Engines with no
internal passes and no reference in the sequence keep ``length = audio``.

Calls flagged cold (the first on a freshly loaded engine, or the first on a
long reference whose passage has not been chosen yet) are kept out of the line
and measured as their own overhead, which the estimate adds only when the
planned render will pay it. Fewer than :data:`MIN_SAMPLES` warm calls: no fit.

**Recency.** A desktop's throughput drifts with what else it is doing: the
same call took 35.7 s and, minutes later under heavier background load, 51.0 s.
The line keeps the shape (what length costs); the machine's current *pace* is
a multiplier on it: the weighted median of the recent calls'
log(observed / predicted), the newest call weighing 1 and each older one half
as much per two calls (``2 ** -(rank / 2)``), with a prior of half a call at
pace 1. By call count, not wall-clock age, because on the measured Mac the
load persisted through an idle hour: forgetting it by age put the next render
37% short, keeping the last calls' level put it 16% short. A median, so one
outlier call cannot move it. Because the level persists, the range widens to
include the long-run line once the newest call is more than 15 minutes old
(the load may have gone either way), and recent calls that scatter widen it
too. Within a running render the live countdown (services/render_countdown.py)
corrects after every call.

Pure math on plain floats: no database, no torch, the same answer on every OS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable, Optional, Sequence

#: The estimate stays silent below this many warm calls for an engine and device.
MIN_SAMPLES = 3
#: Newest warm calls a fit reads. Older rows describe an older driver, torch or
#: thermal state; 120 keeps the pairwise slope set small (7,140 pairs).
FIT_WINDOW = 120
#: Two passes whose lengths differ by less than this carry no slope
#: information: their difference is timing noise divided by almost nothing.
_MIN_DX = 0.25
#: Extra spread for a guess outside what was measured: a steps-scaled fit
#: (compute follows unmasking steps only approximately), a planned pass longer
#: or shorter than any measured one, or a cold start with no overhead measured.
ROUGH_LOW = 0.8
ROUGH_HIGH = 1.25
#: The range never claims better than this, either way: two identical renders
#: on one desktop differ by a few percent from thermal and background load.
_RANGE_FLOOR = 0.10
_CV_FOLDS = 5
#: A planned pass this far outside the measured lengths is an extrapolation.
_EXTRAPOLATION_MARGIN = 0.10
#: Newest cold calls per kind that set its overhead.
_OVERHEAD_WINDOW = 10

#: Recency: each older call's say in the machine's current pace halves every
#: this many calls (chosen on a real drifting Mac, see the module docstring
#: and tests/test_render_fit_recency.py).
PACE_HALF_LIFE_CALLS = 2.0
#: Prior weight (in calls) on "running at the long-run line".
_PACE_PRIOR = 0.5
#: Calls past four half-lives no longer shape the recent spread.
_RECENT_MIN_WEIGHT = 1 / 16
#: After this long without a call, the machine may be back at its long-run
#: speed (or slower still): the range covers both.
IDLE_S = 15 * 60

#: omnivoice/models/omnivoice.py OmniVoiceGenerationConfig defaults.
OMNIVOICE_CHUNK_THRESHOLD_S = 30.0
OMNIVOICE_CHUNK_SECONDS = 15.0


@dataclass(frozen=True)
class CallShape:
    """How an engine turns one call into model passes."""

    #: Audio above which the engine splits a call into passes; None: never.
    chunk_threshold: Optional[float] = None
    chunk_seconds: float = OMNIVOICE_CHUNK_SECONDS
    #: The reference audio is part of every pass's sequence (OmniVoice).
    reference_in_sequence: bool = False

    def passes(self, audio_seconds: float) -> int:
        if self.chunk_threshold is None or audio_seconds <= self.chunk_threshold:
            return 1
        return max(1, math.ceil(audio_seconds / self.chunk_seconds))

    def pass_length(self, audio_seconds: float, ref_seconds: float = 0.0) -> float:
        per_pass = max(0.0, audio_seconds) / self.passes(audio_seconds)
        return per_pass + (max(0.0, ref_seconds or 0.0) if self.reference_in_sequence else 0.0)


PLAIN = CallShape()
OMNIVOICE = CallShape(chunk_threshold=OMNIVOICE_CHUNK_THRESHOLD_S, reference_in_sequence=True)


@dataclass(frozen=True)
class Sample:
    """One completed synthesis call."""

    audio_seconds: float
    wall_seconds: float
    num_step: Optional[int] = None
    ref_seconds: float = 0.0
    #: None for a warm call; "load" or "voice" for a cold one.
    cold: Optional[str] = None
    #: When it finished (epoch seconds); None: its age is unknown.
    created_at: Optional[float] = None


@dataclass(frozen=True)
class Fit:
    """``wall = intercept + slope * length`` for one pass."""

    intercept: float
    slope: float
    low_ratio: float
    high_ratio: float
    samples: int
    rough: bool = False

    def predict(self, length: float) -> float:
        return self.intercept + self.slope * max(0.0, float(length))


@dataclass(frozen=True)
class Overhead:
    """Measured extra seconds one kind of cold call pays."""

    typical: float
    low: float
    high: float
    samples: int


@dataclass(frozen=True)
class CallPrediction:
    seconds: float
    low: float
    high: float
    extrapolated: bool


@dataclass(frozen=True)
class CallModel:
    """What one call costs on this machine, for one engine, device and steps."""

    shape: CallShape
    fit: Fit
    length_min: float
    length_max: float
    overheads: dict = field(default_factory=dict)
    #: The machine's current speed against the line (1.3: 30% slower now).
    pace: float = 1.0
    #: The range, after recent scatter has had its say.
    low_ratio: Optional[float] = None
    high_ratio: Optional[float] = None

    @property
    def rough(self) -> bool:
        return self.fit.rough

    @property
    def samples(self) -> int:
        return self.fit.samples

    def predict(self, audio_seconds: float, ref_seconds: float = 0.0) -> CallPrediction:
        passes = self.shape.passes(audio_seconds)
        length = self.shape.pass_length(audio_seconds, ref_seconds)
        seconds = passes * self.fit.predict(length) * self.pace
        extrapolated = (length < self.length_min * (1 - _EXTRAPOLATION_MARGIN)
                        or length > self.length_max * (1 + _EXTRAPOLATION_MARGIN))
        low_ratio = self.fit.low_ratio if self.low_ratio is None else self.low_ratio
        high_ratio = self.fit.high_ratio if self.high_ratio is None else self.high_ratio
        low, high = seconds * low_ratio, seconds * high_ratio
        if extrapolated:
            low, high = low * ROUGH_LOW, high * ROUGH_HIGH
        return CallPrediction(seconds, low, high, extrapolated)


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


def _line(pts: Sequence[tuple[float, float]]) -> tuple[float, float]:
    slopes = [
        (y2 - y1) / (x2 - x1)
        for i, (x1, y1) in enumerate(pts)
        for x2, y2 in pts[i + 1:]
        if abs(x2 - x1) >= _MIN_DX
    ]
    slope = median(slopes) if slopes else 0.0
    intercept = median(y - slope * x for x, y in pts) if slope > 0 else -1.0
    if slope <= 0 or intercept < 0:
        # No usable spread in length, a negative slope (noise swamping the
        # signal) or a line that would need a negative per-pass cost: time
        # proportional to length is the honest remaining model.
        slope = median(y / x for x, y in pts)
        intercept = 0.0
    return intercept, slope


def _cross_validated_ratios(pts: Sequence[tuple[float, float]]) -> list[float]:
    """observed / predicted for every point, each predicted by a line fitted
    WITHOUT it (k-fold, at most :data:`_CV_FOLDS` folds): the errors a new
    call will actually see, not the smaller in-sample residuals."""
    folds = min(len(pts), _CV_FOLDS)
    ratios: list[float] = []
    for k in range(folds):
        held = [p for i, p in enumerate(pts) if i % folds == k]
        rest = [p for i, p in enumerate(pts) if i % folds != k]
        if len(rest) < MIN_SAMPLES:
            continue
        a, b = _line(rest)
        ratios.extend(y / (a + b * x) for x, y in held if a + b * x > 0)
    return ratios


def fit_line(points: Iterable[tuple[float, float]], *, rough: bool = False) -> Optional[Fit]:
    """Fit ``wall = a + b * length`` robustly; ``None`` below :data:`MIN_SAMPLES`."""
    pts = [(float(x), float(y)) for x, y in points
           if _finite_positive(x) and _finite_positive(y)]
    if len(pts) < MIN_SAMPLES:
        return None
    intercept, slope = _line(pts)
    ratios = _cross_validated_ratios(pts) or [y / (intercept + slope * x) for x, y in pts]
    low = min(1.0 - _RANGE_FLOOR, percentile(ratios, 10))
    high = max(1.0 + _RANGE_FLOOR, percentile(ratios, 90))
    if rough:
        low *= ROUGH_LOW
        high *= ROUGH_HIGH
    return Fit(intercept=intercept, slope=slope, low_ratio=low, high_ratio=high,
               samples=len(pts), rough=rough)


def _step_ratio(sample_step: Optional[int], target_step: Optional[int]) -> float:
    if not sample_step or not target_step:
        return 1.0
    return float(sample_step) / float(target_step)


def _pass_point(sample: Sample, shape: CallShape, scale: float) -> tuple[float, float]:
    passes = shape.passes(sample.audio_seconds)
    return (shape.pass_length(sample.audio_seconds, sample.ref_seconds) * scale,
            sample.wall_seconds / passes)


def _overheads(cold: Sequence[Sample], fit: Fit, shape: CallShape,
               num_step: Optional[int]) -> dict:
    """Extra seconds per cold kind: what each cold call took beyond what the
    warm model says the same call costs."""
    extra: dict = {}
    for s in cold:
        if not (_finite_positive(s.audio_seconds) and _finite_positive(s.wall_seconds)):
            continue
        bucket = extra.setdefault(s.cold, [])
        if len(bucket) >= _OVERHEAD_WINDOW:
            continue
        length, _ = _pass_point(s, shape, _step_ratio(s.num_step, num_step))
        warm = shape.passes(s.audio_seconds) * fit.predict(length)
        bucket.append(max(0.0, s.wall_seconds - warm))
    return {kind: Overhead(typical=median(v), low=min(v), high=max(v), samples=len(v))
            for kind, v in extra.items() if v}


def _weighted_percentile(pairs: Sequence[tuple[float, float]], pct: float) -> float:
    """Percentile of ``(value, weight)`` pairs (weights need not sum to 1)."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    target = total * pct / 100.0
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= target:
            return value
    return ordered[-1][0]


def _recent_pace(used: Sequence[Sample], points: Sequence[tuple[float, float]], fit: Fit,
                 now: Optional[float]) -> tuple[float, Optional[float], Optional[float]]:
    """``(pace, recent_low, recent_high)`` from the newest calls' errors.

    ``used`` is newest first. ``pace`` multiplies the line: exp of the
    weighted median log(observed / predicted), weights halving every
    :data:`PACE_HALF_LIFE_CALLS` calls, plus a prior of half a call at pace 1.
    ``recent_low`` / ``recent_high`` are the 10th / 90th percentile of the
    recent calls' ratios to the paced line, stretched to the long-run line
    when the newest call is older than :data:`IDLE_S`. ``now=None``: no
    recency (pace 1, no recent spread).
    """
    if now is None:
        return 1.0, None, None
    logs: list[tuple[float, float]] = []
    for rank, (x, y) in enumerate(p for p in points
                                  if _finite_positive(p[0]) and _finite_positive(p[1])):
        predicted = fit.predict(x)
        if predicted > 0:
            logs.append((math.log(y / predicted), 2.0 ** (-rank / PACE_HALF_LIFE_CALLS)))
    if not logs:
        return 1.0, None, None
    log_pace = _weighted_percentile([*logs, (0.0, _PACE_PRIOR)], 50)
    pace = math.exp(log_pace)
    recent = [(math.exp(v - log_pace), w) for v, w in logs if w >= _RECENT_MIN_WEIGHT]
    low, high = _weighted_percentile(recent, 10), _weighted_percentile(recent, 90)
    newest = next((s.created_at for s in used if s.created_at is not None), None)
    if newest is not None and now - float(newest) > IDLE_S:
        low, high = min(low, fit.low_ratio / pace), max(high, fit.high_ratio / pace)
    return pace, low, high


def fit_model(samples: Sequence[Sample], num_step: Optional[int],
              shape: CallShape = PLAIN, *, now: Optional[float] = None) -> Optional[CallModel]:
    """The model for calls at ``num_step``, from one engine+device's samples.

    ``samples`` is newest first. Measured when this exact steps bucket has
    :data:`MIN_SAMPLES` warm calls. Otherwise, when the engine and device have
    that many at any steps, every warm call is rescaled to ``num_step`` and the
    fit is marked rough: compute scales with unmasking steps, so a pass at
    ``s`` steps with length ``L`` costs what length ``L * s / num_step`` costs
    at ``num_step``, which scales ``b`` by the steps ratio and leaves the
    per-pass cost ``a`` alone. ``None`` below that: cold start.

    ``now`` (epoch seconds) turns on recency: the machine's current pace from
    the newest calls, and a range widened by their scatter and by idleness.
    """
    warm = [s for s in samples if not s.cold]
    cold = [s for s in samples if s.cold]
    used = [s for s in warm if s.num_step == num_step][:FIT_WINDOW]
    points = [_pass_point(s, shape, 1.0) for s in used]
    fit = fit_line(points)
    if fit is None:
        used = list(warm)[:FIT_WINDOW]
        points = [_pass_point(s, shape, _step_ratio(s.num_step, num_step)) for s in used]
        fit = fit_line(points, rough=True)
    if fit is None:
        return None
    lengths = [x for x, _ in points if _finite_positive(x)]
    pace, recent_low, recent_high = _recent_pace(used, points, fit, now)
    low = fit.low_ratio if recent_low is None else min(fit.low_ratio, recent_low)
    high = fit.high_ratio if recent_high is None else max(fit.high_ratio, recent_high)
    return CallModel(shape=shape, fit=fit, length_min=min(lengths), length_max=max(lengths),
                     overheads=_overheads(cold, fit, shape, num_step), pace=pace,
                     low_ratio=low, high_ratio=high)


def fit_bucket(samples: Sequence[Sample], num_step: Optional[int]) -> Optional[Fit]:
    """The per-call line alone (one pass per call, no reference)."""
    model = fit_model(samples, num_step, PLAIN)
    return model.fit if model is not None else None


__all__ = [
    "FIT_WINDOW", "IDLE_S", "MIN_SAMPLES", "OMNIVOICE", "PACE_HALF_LIFE_CALLS", "PLAIN",
    "ROUGH_HIGH", "ROUGH_LOW",
    "CallModel", "CallPrediction", "CallShape", "Fit", "Overhead", "Sample",
    "fit_bucket", "fit_line", "fit_model", "percentile",
]
