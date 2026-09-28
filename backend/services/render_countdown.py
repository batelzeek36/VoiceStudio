"""Seconds left in a running longform render, re-fitted after every engine call.

docs/adr/render-time-estimate.md. At the start of a render the estimate plans
every engine call of every chapter (the same planning code
``POST /render/estimate`` runs) and prices each one. While the render runs,
``synthesize_chapter`` reports each call as it finishes, with its position in
that plan and its wall time, and each span its segment cache served instead.

The countdown is what the plan says the unfinished calls cost, times the pace
so far: actual / predicted over the calls rendered in THIS render, clamped to
0.5x..3x and applied from the second call on (one call is too noisy a sample,
and the first can carry a model load). So a one-chapter book is corrected
call by call instead of only when it ends, and a machine that slows down
mid-render (another heavy job starts) pulls the countdown up within a call or
two. Cached calls cost nothing and are never counted as work left.
"""
from __future__ import annotations

import threading
from typing import Optional, Sequence

#: Calls rendered before the pace moves the countdown.
MIN_CALLS = 2
PACE_MIN = 0.5
PACE_MAX = 3.0


class RenderCountdown:
    """Thread-safe: calls are reported from the GPU worker, read from the stream."""

    def __init__(self, plan: Sequence[Sequence[float]]):
        self._plan = [[max(0.0, float(s)) for s in chapter] for chapter in plan]
        self._finished = [set() for _ in self._plan]
        self._actual = 0.0
        self._predicted = 0.0
        self._rendered = 0
        self._lock = threading.Lock()

    @property
    def total_calls(self) -> int:
        return sum(len(chapter) for chapter in self._plan)

    def _valid(self, chapter: int, position: int) -> bool:
        return 0 <= chapter < len(self._plan) and 0 <= position < len(self._plan[chapter])

    def call_done(self, chapter: int, position: int, seconds: Optional[float]) -> None:
        """A call finished: ``seconds`` of wall time, or None when the segment
        cache served it (no time spent, nothing to learn from)."""
        with self._lock:
            if not self._valid(chapter, position) or position in self._finished[chapter]:
                return
            self._finished[chapter].add(position)
            if seconds is not None and seconds >= 0:
                self._actual += float(seconds)
                self._predicted += self._plan[chapter][position]
                self._rendered += 1

    def chapter_done(self, chapter: int) -> None:
        """A chapter ended (rendered, served from the chapter cache, or failed):
        none of its calls is work left."""
        with self._lock:
            if 0 <= chapter < len(self._plan):
                self._finished[chapter] = set(range(len(self._plan[chapter])))

    def _pace(self) -> float:
        if self._rendered < MIN_CALLS or self._predicted <= 0:
            return 1.0
        return min(PACE_MAX, max(PACE_MIN, self._actual / self._predicted))

    @property
    def pace(self) -> float:
        with self._lock:
            return self._pace()

    @property
    def calls_done(self) -> int:
        with self._lock:
            return sum(len(done) for done in self._finished)

    def remaining(self) -> tuple[float, float]:
        """``(seconds_left, next_call_seconds)``: the paced cost of every
        unfinished call, and of the first of them (the one in flight), so a
        client can count the in-flight call down without going below the calls
        that have not started."""
        with self._lock:
            pace = self._pace()
            left = 0.0
            first: Optional[float] = None
            for chapter, done in zip(self._plan, self._finished):
                for position, seconds in enumerate(chapter):
                    if position in done:
                        continue
                    left += seconds
                    first = seconds if first is None else first
            return pace * left, pace * (first or 0.0)


__all__ = ["MIN_CALLS", "PACE_MAX", "PACE_MIN", "RenderCountdown"]
