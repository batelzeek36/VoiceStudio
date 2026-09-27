"""The render-time model: robust, clamped, honest about cold starts.

docs/adr/render-time-estimate.md. Synthetic samples with a known per-call cost
``a`` and per-audio-second cost ``b``, plus noise and outliers, must come back
as that line; a machine with fewer than three samples must get no number.
"""
import random

import pytest

from services.render_fit import (
    MIN_SAMPLES,
    Sample,
    fit_bucket,
    fit_line,
    percentile,
)


def _line(a, b, xs, *, noise=0.0, seed=7):
    rng = random.Random(seed)
    return [(x, a + b * x + rng.uniform(-noise, noise)) for x in xs]


def test_recovers_known_line_from_noisy_samples():
    xs = [2 + 0.5 * i for i in range(40)]
    fit = fit_line(_line(3.0, 4.2, xs, noise=0.4))
    assert fit.intercept == pytest.approx(3.0, abs=0.5)
    assert fit.slope == pytest.approx(4.2, abs=0.08)
    assert fit.samples == 40
    assert fit.low_ratio <= 1.0 <= fit.high_ratio
    # A 66 s read at a known 3 + 4.2 s/s line.
    assert fit.predict(66) == pytest.approx(3 + 4.2 * 66, rel=0.03)


def test_outliers_do_not_move_the_fit():
    """A warm-up call and a stall (the classic first-call compile) are ignored."""
    points = _line(1.0, 2.0, [1 + i for i in range(20)], noise=0.1)
    points[0] = (points[0][0], 90.0)     # first call paid a model load
    points[7] = (points[7][0], 400.0)    # raced another job
    points[13] = (points[13][0], 0.05)   # clock glitch
    fit = fit_line(points)
    assert fit.slope == pytest.approx(2.0, abs=0.1)
    assert fit.intercept == pytest.approx(1.0, abs=0.5)


def test_clamps_negative_intercept_to_a_proportional_model():
    # Costs a little LESS than proportional at short lengths: a < 0 would
    # predict negative time for a short call.
    fit = fit_line([(1.0, 1.5), (2.0, 4.0), (4.0, 9.0), (8.0, 19.0)])
    assert fit.intercept == 0.0
    assert fit.slope > 0
    assert fit.predict(0.1) > 0


def test_identical_lengths_fall_back_to_wall_per_audio_second():
    """The same sentence rendered three times carries no slope information."""
    fit = fit_line([(5.0, 20.0), (5.0, 22.0), (5.0, 21.0)])
    assert fit.intercept == 0.0
    assert fit.slope == pytest.approx(21.0 / 5.0)


def test_negative_slope_from_noise_is_clamped():
    fit = fit_line([(1.0, 10.0), (2.0, 9.5), (3.0, 9.0), (4.0, 8.9)])
    assert fit.slope > 0 and fit.intercept >= 0


def test_cold_start_below_three_samples_is_no_number():
    assert fit_line([]) is None
    assert fit_line([(1.0, 2.0), (2.0, 4.0)]) is None
    assert MIN_SAMPLES == 3
    # Unusable rows (no audio, non-finite) do not count toward the three.
    assert fit_line([(1.0, 2.0), (2.0, 4.0), (0.0, 1.0), (float("nan"), 3.0)]) is None


def test_bucket_prefers_its_own_steps_and_marks_it_measured():
    samples = [Sample(x, 1 + 2 * x, 32) for x in (2, 4, 6, 8)] + [
        Sample(x, 1 + 1 * x, 16) for x in (2, 4, 6, 8)
    ]
    fit = fit_bucket(samples, 32)
    assert not fit.rough
    assert fit.slope == pytest.approx(2.0)


def test_other_steps_scale_the_slope_not_the_per_call_cost():
    """Measured at 16 steps, asked about 32: b doubles, a stays (rough)."""
    samples = [Sample(x, 5 + 3 * x, 16) for x in (2, 4, 6, 8, 10)]
    fit = fit_bucket(samples, 32)
    assert fit.rough
    assert fit.intercept == pytest.approx(5.0)
    assert fit.slope == pytest.approx(6.0)
    # The rough range is wider than the measured one.
    measured = fit_bucket(samples, 16)
    assert fit.low_ratio < measured.low_ratio
    assert fit.high_ratio > measured.high_ratio


def test_pooled_rough_fit_uses_every_steps_bucket():
    samples = [Sample(4.0, 2 + 4 * 1.0, 16), Sample(8.0, 2 + 8 * 1.0, 16),
               Sample(4.0, 2 + 4 * 2.0, 32)]
    fit = fit_bucket(samples, 64)
    assert fit.rough and fit.samples == 3
    assert fit.slope == pytest.approx(4.0, rel=0.05)


def test_engine_without_steps_uses_its_single_bucket():
    samples = [Sample(x, 0.5 + 0.2 * x, None) for x in (3, 6, 9)]
    fit = fit_bucket(samples, None)
    assert not fit.rough
    assert fit.slope == pytest.approx(0.2)


def test_fewer_than_three_samples_for_the_device_is_cold_start():
    assert fit_bucket([Sample(2.0, 4.0, 16), Sample(4.0, 7.0, 32)], 16) is None


def test_fit_reads_only_the_newest_window():
    old = [Sample(x, 100 * x, 16) for x in (1, 2, 3)]
    new = [Sample(x, 2 * x, 16) for x in range(1, 121)]
    fit = fit_bucket(new + old, 16)  # newest first
    assert fit.slope == pytest.approx(2.0)


def test_percentile_matches_linear_interpolation():
    assert percentile([1, 2, 3, 4], 25) == pytest.approx(1.75)
    assert percentile([1, 2, 3, 4], 75) == pytest.approx(3.25)
    assert percentile([5], 50) == 5
    with pytest.raises(ValueError):
        percentile([], 50)
