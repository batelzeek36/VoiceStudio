"""Recency: the estimate follows the machine's current speed.

A desktop's throughput drifts with background load. The model keeps its line
(what length costs) and multiplies it by the machine's current pace: the
weighted median of the newest calls' errors, weights halving every two calls,
so the level persists across idle time (on the measured Mac the load did) and
one outlier cannot move it. The range widens when recent calls scatter, and
covers the long-run line once the newest call is more than 15 minutes old.
"""
import json
import random
from pathlib import Path
from statistics import median

import pytest

from services.render_fit import (
    IDLE_S, OMNIVOICE, PACE_HALF_LIFE_CALLS, PLAIN, Sample, fit_model,
)

_DRIFT = Path(__file__).parent / "fixtures" / "render_timings_m1max_drift.json"
_BASE = 4 + 1.5 * 15.0  # what a 15 s call costs at the machine's usual speed


def _shifted_machine(shift=1.4, calls=40, every=60.0, seed=3):
    """``calls`` calls one per ``every`` seconds; halfway the machine slows by ``shift``."""
    rng = random.Random(seed)
    samples = []
    for i in range(calls):
        audio = rng.uniform(3, 25)
        pace = 1.0 if i < calls // 2 else shift
        wall = pace * (4 + 1.5 * audio) * rng.uniform(0.95, 1.05)
        samples.append(Sample(audio, wall, 32, 0.0, None, created_at=i * every))
    return samples


def _model(history, *, idle=1.0):
    newest_first = history[::-1]
    return fit_model(newest_first, 32, PLAIN, now=newest_first[0].created_at + idle)


def test_a_forty_percent_slowdown_is_tracked_within_a_few_calls():
    samples = _shifted_machine()
    for seen in (23, 26, 30, 40):  # 3, 6, 10 and 20 calls into the slow regime
        tracked = _model(samples[:seen]).predict(15.0).seconds
        untracked = fit_model(samples[:seen][::-1], 32, PLAIN).predict(15.0).seconds
        assert tracked == pytest.approx(1.4 * _BASE, rel=0.10)
        assert abs(untracked / (1.4 * _BASE) - 1) > abs(tracked / (1.4 * _BASE) - 1)


def test_the_same_machine_back_at_full_speed_is_tracked_too():
    samples = _shifted_machine(shift=1 / 1.4)
    assert _model(samples[:26]).predict(15.0).seconds == pytest.approx(_BASE / 1.4, rel=0.10)


def test_one_outlier_call_does_not_move_the_pace():
    samples = _shifted_machine(shift=1.0)
    spike = samples[-1]
    samples[-1] = Sample(spike.audio_seconds, spike.wall_seconds * 3, 32, 0.0, None,
                         spike.created_at)
    assert _model(samples).pace == pytest.approx(1.0, abs=0.06)


def test_the_last_level_persists_through_idle_time_and_the_range_covers_both():
    samples = _shifted_machine()
    fresh = _model(samples[:30]).predict(15.0)
    idle = _model(samples[:30], idle=IDLE_S + 3600).predict(15.0)
    # Still the slow level last seen (the load may well still be there) ...
    assert idle.seconds == pytest.approx(fresh.seconds)
    # ... but after an idle stretch the range also covers the usual speed.
    assert fresh.low > _BASE
    assert idle.low <= _BASE * 0.95
    assert PACE_HALF_LIFE_CALLS == 2.0


def test_the_range_widens_when_recent_calls_scatter():
    steady = _shifted_machine(shift=1.0)
    erratic = [Sample(s.audio_seconds, s.wall_seconds * (1.5 if i % 2 else 0.7), 32, 0.0,
                      None, s.created_at) if i >= 34 else s
               for i, s in enumerate(steady)]
    calm = _model(steady).predict(15.0)
    wild = _model(erratic).predict(15.0)
    assert wild.high / wild.seconds > calm.high / calm.seconds + 0.2
    assert wild.low / wild.seconds < calm.low / calm.seconds


def test_without_now_there_is_no_recency():
    samples = _shifted_machine()
    assert fit_model(samples[::-1], 32, PLAIN).pace == 1.0


@pytest.fixture(scope="module")
def drift():
    rows = json.loads(_DRIFT.read_text(encoding="utf-8"))["rows"]
    return [dict(r, sample=Sample(r["audio_seconds"], r["wall_seconds"], r["num_step"],
                                  r["ref_seconds"] or 0.0, r["cold"], r["t"])) for r in rows]


def _history(rows, started):
    return [r["sample"] for r in rows if r["t"] <= started][::-1]


def _one_step_ahead(rows, *, recency):
    """Estimate each warm 64-step call from the calls finished before it began."""
    out = []
    for row in rows:
        if row["cold"] or row["num_step"] != 64:
            continue
        started = row["t"] - row["wall_seconds"]
        model = fit_model(_history(rows, started), 64, OMNIVOICE,
                          now=started if recency else None)
        if model is not None:
            out.append((row, model.predict(row["audio_seconds"], row["ref_seconds"])))
    return out


def test_on_the_real_drifting_mac_recency_cuts_the_error_after_the_shift(drift):
    with_recency = _one_step_ahead(drift, recency=True)
    without = _one_step_ahead(drift, recency=False)

    def after_shift(results):
        return median(abs(p.seconds / r["wall_seconds"] - 1) for r, p in results
                      if r["t"] >= 2400)

    def inside(results):
        return sum(p.low <= r["wall_seconds"] <= p.high for r, p in results)

    assert after_shift(without) > 0.30
    assert after_shift(with_recency) < 0.12
    assert inside(with_recency) > inside(without)


def test_a_render_after_an_idle_hour_keeps_the_machines_last_known_pace(drift):
    """The 6-call render started 50 minutes after the last call, load still high."""
    render = drift[-6:]
    started = render[0]["t"] - render[0]["wall_seconds"]
    model = fit_model(_history(drift, started), 64, OMNIVOICE, now=started)
    predicted = sum(model.predict(r["audio_seconds"], r["ref_seconds"]).seconds for r in render)
    predicted += model.overheads["load"].typical  # the sidecar had been reaped
    actual = sum(r["wall_seconds"] for r in render)
    assert predicted == pytest.approx(actual, rel=0.20)
    assert model.pace > 1.3
