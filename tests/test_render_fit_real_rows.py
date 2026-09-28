"""The render-time model against real calls from one machine
(tests/fixtures/render_timings_m1max_omnivoice_mps.json, numbers only).

A single line on audio seconds missed these by a median 26%: OmniVoice splits a
call over 30 s into ~15 s passes that each pay the fixed cost, and every pass
carries the voice's reference audio in its sequence. The model prices passes
of (audio + reference), keeps cold calls out of the line and measures them as
their own overhead.
"""
import json
from pathlib import Path
from statistics import median

import pytest

from services.render_fit import OMNIVOICE, PLAIN, Sample, fit_model

_FIXTURE = Path(__file__).parent / "fixtures" / "render_timings_m1max_omnivoice_mps.json"


@pytest.fixture(scope="module")
def real():
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    samples = [Sample(r["audio_seconds"], r["wall_seconds"], data["num_step"],
                      r["ref_seconds"], r["cold"]) for r in data["rows"]]
    return data["num_step"], samples  # oldest first, as recorded


def _leave_one_out(samples, num_step, shape):
    """Each warm call predicted by a model fitted on every other call."""
    results = []
    for i, target in enumerate(samples):
        if target.cold:
            continue
        others = [s for j, s in enumerate(samples) if j != i][::-1]  # newest first
        model = fit_model(others, num_step, shape)
        results.append((target, model.predict(target.audio_seconds, target.ref_seconds)))
    return results


def test_leave_one_out_meets_the_bar(real):
    num_step, samples = real
    results = _leave_one_out(samples, num_step, OMNIVOICE)
    assert len(results) == 7
    errors = [abs(p.seconds / t.wall_seconds - 1) for t, p in results]
    inside = sum(p.low <= t.wall_seconds <= p.high for t, p in results)
    assert median(errors) <= 0.20
    assert inside >= 5


def test_a_single_line_on_audio_seconds_does_not(real):
    """The model this replaces: one line on audio length, no passes or reference."""
    num_step, samples = real
    results = _leave_one_out(samples, num_step, PLAIN)
    errors = [abs(p.seconds / t.wall_seconds - 1) for t, p in results]
    assert median(errors) > 0.20


def test_the_longer_reference_is_cost_not_a_cold_start(real):
    """The first call on the 14.95 s voice looked 'cold-ish'; its sequence
    (reference + audio) matches a warm call on the 11.35 s voice."""
    num_step, samples = real
    new_voice = samples[4]
    model = fit_model([s for s in samples if s is not new_voice][::-1], num_step, OMNIVOICE)
    predicted = model.predict(new_voice.audio_seconds, new_voice.ref_seconds).seconds
    assert predicted == pytest.approx(new_voice.wall_seconds, rel=0.10)


def test_cold_calls_stay_out_of_the_line_and_become_the_load_overhead(real):
    num_step, samples = real
    with_cold = fit_model(samples[::-1], num_step, OMNIVOICE)
    without = fit_model([s for s in samples if not s.cold][::-1], num_step, OMNIVOICE)
    assert (with_cold.fit.intercept, with_cold.fit.slope) == (without.fit.intercept, without.fit.slope)
    assert with_cold.samples == 7
    load = with_cold.overheads["load"]
    first = samples[0]
    warm_cost = with_cold.predict(first.audio_seconds, first.ref_seconds).seconds
    assert load.typical == pytest.approx(first.wall_seconds - warm_cost)
    assert 3 < load.typical < first.wall_seconds


def test_long_calls_are_priced_as_passes(real):
    num_step, samples = real
    model = fit_model(samples[::-1], num_step, OMNIVOICE)
    # 45 s of audio is three ~15 s passes, not one 45 s pass.
    three = model.predict(45.0, 11.35).seconds
    assert three == pytest.approx(3 * model.predict(15.0, 11.35).seconds)


def test_planned_calls_outside_the_measured_lengths_are_rough(real):
    num_step, samples = real
    model = fit_model(samples[::-1], num_step, OMNIVOICE)
    inside = model.predict(20.0, 11.35)
    shorter = model.predict(1.0, 0.0)
    assert not inside.extrapolated
    assert shorter.extrapolated
    assert shorter.high / shorter.seconds > inside.high / inside.seconds
