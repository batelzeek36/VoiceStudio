"""The live countdown: re-fitted after every engine call, not every chapter.

A one-chapter book used to get no correction until it ended. The render now
reports each call as it finishes; the countdown rescales the work left by
actual / predicted over the calls done so far (from the second call, clamped
to 0.5x..3x), and cached calls are never work left.
"""
import asyncio
import json
import types

import pytest
import torch

from services.render_countdown import MIN_CALLS, PACE_MAX, PACE_MIN, RenderCountdown


def test_counts_the_plan_down_before_the_pace_is_known():
    c = RenderCountdown([[10, 20, 30]])
    assert c.remaining() == (60, 10)
    c.call_done(0, 0, 25.0)  # one slow call is not yet a pace
    assert MIN_CALLS == 2
    assert c.pace == 1.0
    assert c.remaining() == (50, 20)


def test_rescales_what_is_left_by_actual_over_predicted():
    c = RenderCountdown([[10, 20, 30, 40]])
    c.call_done(0, 0, 15.0)
    c.call_done(0, 1, 30.0)  # 45 s for 30 s of plan: 1.5x
    assert c.pace == pytest.approx(1.5)
    left, current = c.remaining()
    assert left == pytest.approx(1.5 * 70)
    assert current == pytest.approx(1.5 * 30)


def test_the_pace_is_clamped():
    slow = RenderCountdown([[1, 1, 1]])
    slow.call_done(0, 0, 50.0)
    slow.call_done(0, 1, 50.0)
    assert slow.pace == PACE_MAX == 3.0
    fast = RenderCountdown([[10, 10, 10]])
    fast.call_done(0, 0, 0.1)
    fast.call_done(0, 1, 0.1)
    assert fast.pace == PACE_MIN == 0.5


def test_cached_calls_and_finished_chapters_are_not_work_left():
    c = RenderCountdown([[10, 10], [20, 20]])
    c.call_done(0, 0, None)  # served by the segment cache
    c.call_done(0, 1, 10.0)
    assert c.pace == 1.0  # a cached call teaches nothing about speed
    c.chapter_done(1)  # a whole chapter from the chapter cache
    assert c.remaining() == (0, 0)
    assert c.calls_done == c.total_calls == 4


def test_reports_outside_the_plan_are_ignored():
    c = RenderCountdown([[10]])
    c.call_done(3, 0, 5.0)
    c.call_done(0, 7, 5.0)
    c.call_done(0, 0, 5.0)
    c.call_done(0, 0, 500.0)  # a repeat never counts twice
    assert c.remaining() == (0, 0) and c.calls_done == 1


def test_synthesize_chapter_reports_calls_in_plan_order(monkeypatch):
    """Positions match the planner's list, including spans the cache served."""
    from services import render_plan
    from services.audiobook import Span, normalized_spans, synthesize_chapter

    spans = normalized_spans([
        Span(voice_id=None, text="First line."),
        Span(voice_id="b", text="Cached line.", speed=1.2),
        Span(voice_id=None, text=" ".join(["A long sentence that fills a chunk."] * 40)),
    ], "English")
    planned = render_plan.chapter_calls(spans, lexicon=None, paragraph_gap_ms=0)

    class Cache:
        hits = misses = 0

        def load(self, span, nonce=0):
            return torch.zeros(1, 100) if span.text == "Cached line." else None

        def store(self, span, audio, nonce=0):
            pass

    reports = []
    synthesize_chapter(spans, lambda *a: torch.zeros(1, 100), 1000, crossfade_ms=0,
                       segment_cache=Cache(), on_call=lambda p, s: reports.append((p, s)))
    assert [p for p, _ in reports] == list(range(len(planned)))
    assert [s is None for _, s in reports] == [c.text == "Cached line." for c in planned]


def test_a_broken_listener_never_breaks_the_chapter():
    from services.audiobook import Span, synthesize_chapter

    def boom(position, seconds):
        raise RuntimeError("listener fell over")

    audio, _ = synthesize_chapter([Span(voice_id=None, text="Hello.")],
                                  lambda *a: torch.zeros(1, 100), 1000, crossfade_ms=0,
                                  on_call=boom)
    assert audio.shape[-1] == 100


def test_a_one_chapter_render_streams_a_corrected_countdown_after_each_call(
        tmp_path, monkeypatch):
    """Drives the real SSE generator: 4 planned calls of 10 s that really take 20 s."""
    from services.ffmpeg_utils import find_ffmpeg

    if find_ffmpeg() is None:
        pytest.skip("ffmpeg required for the longform render")
    from api.routers import audiobook
    from api.routers import render_estimate
    from services import audiobook as audiobook_service
    from services.audiobook import AudiobookPlan, Chapter, Span

    clock = {"t": 0.0}
    monkeypatch.setattr(audiobook_service, "time",
                        types.SimpleNamespace(perf_counter=lambda: clock["t"]))

    def synth(text, voice_id, speed=None):
        clock["t"] += 20.0  # every call takes twice its plan
        return torch.zeros(2400)

    monkeypatch.setattr(audiobook, "_build_synth", lambda *a, **k: {
        "mode": "generic", "engine_id": "stub", "synth": synth, "sample_rate": 24000,
        "resolve": lambda _v: {"ref_audio": None, "ref_text": None, "instruct": None,
                               "seed": None}})
    monkeypatch.setattr(render_estimate, "plan_longform", lambda plan, **kw: {
        "seconds": 40.0, "parts": [{"calls": 4, "seconds": 40.0,
                                    "call_seconds": [10.0, 10.0, 10.0, 10.0]}]})
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path))
    plan = AudiobookPlan(chapters=[Chapter(title="Only", spans=[
        Span(voice_id=None, text=f"Line number {n}.") for n in range(4)])])

    async def run():
        return [json.loads(f[len("data:"):]) async for f in
                audiobook._render_longform_sse(plan, default_voice=None, fmt="mp3")]

    events = asyncio.run(asyncio.wait_for(run(), timeout=120))
    progress = [e for e in events if e["type"] == "progress"]
    # At start, then after each of the four calls, then once the chapter ends.
    assert [p["calls_done"] for p in progress] == [0, 1, 2, 3, 4, 4]
    assert [p["remaining_s"] for p in progress] == [40.0, 30.0, 40.0, 20.0, 0.0, 0.0]
    assert progress[2]["pace"] == 2.0 and progress[2]["next_call_s"] == 20.0
    assert [e["type"] for e in events if e["type"] != "progress"] == [
        "started", "chapter", "assembling", "done"]


def test_closing_the_stream_cancels_the_chapter():
    from services.longform_progress import chapter_with_progress

    started = {}

    async def slow_chapter(on_call):
        on_call(0, 1.0)
        started["task"] = asyncio.current_task()
        await asyncio.sleep(60)

    async def run():
        steps = chapter_with_progress(slow_chapter, RenderCountdown([[1.0, 1.0]]), 0)
        kind, event = await steps.__anext__()
        assert kind == "progress" and event["calls_done"] == 1
        await steps.aclose()  # the client went away mid-chapter
        await asyncio.sleep(0)
        return started["task"]

    task = asyncio.run(run())
    assert task.cancelled()
