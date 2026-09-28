"""Live render-time progress for a chapterized render (docs/adr/render-time-estimate.md).

A chapter renders as one GPU job, so the SSE stream used to hear nothing
between chapter events: a one-chapter book showed no correction until it
ended. :func:`chapter_with_progress` runs one chapter while draining the
per-call reports ``synthesize_chapter`` makes from the worker thread, feeds
them to the :class:`services.render_countdown.RenderCountdown`, and yields a
progress payload after each call, then the chapter's result.

A chapter sent to a remote worker is one task that reports nothing until it
returns, so :func:`timed_chapter` feeds the countdown one call per chapter:
the pace is re-fitted per chapter, and between chapters the client counts the
chapter in flight down from its planned cost (estimate minus elapsed).
"""
from __future__ import annotations

import asyncio
import logging
import queue
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from services.render_countdown import RenderCountdown

logger = logging.getLogger("omnivoice.longform_progress")

#: How often the stream looks for finished calls while a chapter renders.
POLL_S = 0.25

OnCall = Callable[[int, Optional[float]], None]


def countdown_for(estimate: Optional[dict]) -> Optional[RenderCountdown]:
    """A countdown from a detailed estimate, or None when it has no number."""
    if not estimate or estimate.get("seconds") is None:
        return None
    plan = [part.get("call_seconds") for part in estimate.get("parts") or []]
    if not plan or any(not isinstance(p, list) for p in plan):
        return None
    return RenderCountdown(plan)


def progress_event(countdown: RenderCountdown, chapter: int) -> dict:
    """The additive ``progress`` SSE event: seconds left (paced), the paced
    cost of the call in flight, and how far through the planned calls it is."""
    left, next_call = countdown.remaining()
    return {
        "type": "progress", "index": chapter,
        "calls_done": countdown.calls_done, "calls": countdown.total_calls,
        "remaining_s": round(left, 1), "next_call_s": round(next_call, 1),
        "pace": round(countdown.pace, 2),
    }


async def timed_chapter(
    start: Callable[[], Awaitable[Any]],
    countdown: RenderCountdown,
    chapter: int,
) -> Any:
    """Run a chapter that reports no calls of its own (a remote worker renders
    it as one task) and tell ``countdown`` what it cost as its single call.
    A chapter served from the cache (``result[2]``) cost nothing and teaches
    the pace nothing; a failed one propagates and is closed by the caller."""
    started = time.perf_counter()
    result = await start()
    cached = bool(result[2]) if isinstance(result, tuple) and len(result) > 2 else False
    countdown.call_done(chapter, 0, None if cached else time.perf_counter() - started)
    return result


async def chapter_with_progress(
    start: Callable[[OnCall], Awaitable[Any]],
    countdown: RenderCountdown,
    chapter: int,
) -> AsyncIterator[tuple[str, Any]]:
    """Run ``start(on_call)``; yield ``("progress", event)`` after every call it
    reports, then ``("result", value)``. The chapter's own exception propagates
    from the final step. Closing or cancelling this generator cancels the
    chapter, exactly like cancelling a plain ``await`` of it."""
    reports: "queue.SimpleQueue[tuple[int, Optional[float]]]" = queue.SimpleQueue()
    task = asyncio.ensure_future(start(lambda position, seconds: reports.put((position, seconds))))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=POLL_S)
            while True:
                try:
                    position, seconds = reports.get_nowait()
                except queue.Empty:
                    break
                # One frame per call, in order, even when several finished
                # between two polls: each is a reading after that call.
                countdown.call_done(chapter, position, seconds)
                yield "progress", progress_event(countdown, chapter)
            if done:
                break
        yield "result", task.result()
    finally:
        if not task.done():
            task.cancel()


__all__ = ["POLL_S", "chapter_with_progress", "countdown_for", "progress_event", "timed_chapter"]
