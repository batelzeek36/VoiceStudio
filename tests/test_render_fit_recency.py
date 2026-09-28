"""Recency: the estimate follows the machine's current speed.

A desktop's throughput drifts with background load. The model keeps its line
(what length costs) and multiplies it by the machine's current pace, taken from
the recent calls weighted by wall-clock age (half-life 2 minutes), and widens
the range when the recent calls scatter.
"""
import json
import random
from pathlib import Path
from statistics import median

import pytest

from services.render_fit import OMNIVOICE, PACE_HALF_LIFE_S, PLAIN, Sample, fit_model

_DRIFT = Path(__file__).parent / "fixtures" / "render_timings_m1max_drift.json"


def _shifted_machine(shift=1.4, calls=40, every=60.0, seed=3):
    """``calls`` calls one per ``every`` seconds; halfway the machine slows by ``shift``."""
    rng = random.Random(seed)
    samples, truth = [], []
    for i in range(calls):
        audio = rng.uniform(3, 25)
        pace = 1.0 if i < calls // 2 else shift
        wall = pace * (4 + 1.5 * audio) * rng.uniform(0.95, 1.05)
        samples.append(Sample(audio, wall, 32, 0.0, None, created_at=i * every))
        truth.append(pace)
    return samples, truth


def _cost(model, audio=15.0):
    return model.predict(audio).seconds


def test_a_forty_percent_slowdown_is_tracked_within_a_few_calls():
    samples, _ = _shifted_machine()
    for seen in (26, 30, 40):  # 6, 10 and 20 calls (minutes) into the slow regime
        history = samples[:seen][::-1]
        now = history[0].created_at + 1
        tracked = _cost(fit_model(history, 32, PLAIN, now=now))
        untracked = _cost(fit_model(history, 32, PLAIN))
        slow_truth = 1.4 * (4 + 1.5 * 15.0)
        assert tracked == pytest.approx(slow_truth, rel=0.10)
        assert abs(untracked / slow_truth - 1) > abs(tracked / slow_truth - 1)


def test_the_same_machine_back_at_full_speed_is_tracked_too():
    samples, _ = _shifted_machine(shift=1 / 1.4)
    history = samples[:30][::-1]
    model = fit_model(history, 32, PLAIN, now=history[0].created_at + 1)
    assert _cost(model) == pytest.approx((4 + 1.5 * 15.0) / 1.4, rel=0.10)


def test_after_an_idle_hour_the_old_evidence_ages_out():
    samples, _ = _shifted_machine()
    history = samples[:30][::-1]
    fresh = fit_model(history, 32, PLAIN, now=history[0].created_at + 1)
    idle = fit_model(history, 32, PLAIN, now=history[0].created_at + 3600)
    assert fresh.pace > 1.2
    # Thirty half-lives later the last session says nothing about now:
    # back to the machine's long-run line.
    assert idle.pace == pytest.approx(1.0, abs=0.01)
    assert PACE_HALF_LIFE_S == 120.0


def test_the_range_widens_when_recent_calls_scatter():
    steady, _ = _shifted_machine(shift=1.0)
    erratic = [Sample(s.audio_seconds, s.wall_seconds * (1.5 if i % 2 else 0.7), 32, 0.0,
                      None, s.created_at) if i >= 34 else s
               for i, s in enumerate(steady)]
    now = steady[-1].created_at + 1
    calm = fit_model(steady[::-1], 32, PLAIN, now=now).predict(15.0)
    wild = fit_model(erratic[::-1], 32, PLAIN, now=now).predict(15.0)
    assert wild.high / wild.seconds > calm.high / calm.seconds + 0.2
    assert wild.low / wild.seconds < calm.low / calm.seconds


def test_without_ages_the_model_is_unchanged():
    samples, _ = _shifted_machine()
    for s in samples:
        assert s.created_at is not None
    ageless = [Sample(s.audio_seconds, s.wall_seconds, s.num_step) for s in samples][::-1]
    assert fit_model(ageless, 32, PLAIN, now=1e9).pace == 1.0


@pytest.fixture(scope="module")
def drift():
    rows = json.loads(_DRIFT.read_text(encoding="utf-8"))["rows"]
    return [dict(r, sample=Sample(r["audio_seconds"], r["wall_seconds"], r["num_step"],
                                  r["ref_seconds"] or 0.0, r["cold"], r["t"])) for r in rows]


def _one_step_ahead(rows, *, recency):
    """Estimate each warm 64-step call from the calls finished before it began."""
    out = []
    for row in rows:
        if row["cold"] or row["num_step"] != 64:
            continue
        started = row["t"] - row["wall_seconds"]
        history = [r["sample"] for r in rows if r["t"] <= started][::-1]
        model = fit_model(history, 64, OMNIVOICE, now=started if recency else None)
        if model is None:
            continue
        p = model.predict(row["audio_seconds"], row["ref_seconds"])
        out.append((row, p))
    return out


def test_on_the_real_drifting_mac_recency_halves_the_error_after_the_shift(drift):
    with_recency = _one_step_ahead(drift, recency=True)
    without = _one_step_ahead(drift, recency=False)

    def after_shift(results):
        return median(abs(p.seconds / r["wall_seconds"] - 1) for r, p in results if r["t"] >= 2400)

    def inside(results):
        return sum(p.low <= r["wall_seconds"] <= p.high for r, p in results)

    assert after_shift(without) > 0.30
    assert after_shift(with_recency) < 0.20
    assert inside(with_recency) >= inside(without)
