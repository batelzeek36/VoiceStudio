/**
 * Render-time estimate display maths (docs/adr/render-time-estimate.md).
 * Pure functions: no React, no clock of their own.
 */

/**
 * Round an estimate to the precision it can honestly claim: 5 s steps under a
 * minute, whole minutes under an hour, 5-minute steps under ten hours, whole
 * hours beyond. Never zero for a positive estimate.
 */
export function roundEstimateSeconds(seconds: number): number {
  if (!Number.isFinite(seconds) || seconds <= 0) return 0;
  if (seconds < 60) return Math.max(5, Math.round(seconds / 5) * 5);
  if (seconds < 3600) return Math.max(60, Math.round(seconds / 60) * 60);
  if (seconds < 36_000) return Math.round(seconds / 300) * 300;
  return Math.round(seconds / 3600) * 3600;
}

function unit(locale: string, value: number, name: 'second' | 'minute' | 'hour'): string {
  return new Intl.NumberFormat(locale, {
    style: 'unit',
    unit: name,
    unitDisplay: 'short',
    maximumFractionDigits: 0,
  }).format(value);
}

/** "45 sec", "4 min", "1 hr 25 min" in the UI locale, via Intl (no strings to translate). */
export function formatRenderDuration(seconds: number, locale: string): string {
  const rounded = roundEstimateSeconds(seconds);
  if (rounded < 60) return unit(locale, rounded, 'second');
  const minutes = Math.round(rounded / 60);
  if (minutes < 60) return unit(locale, minutes, 'minute');
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest
    ? `${unit(locale, hours, 'hour')} ${unit(locale, rest, 'minute')}`
    : unit(locale, hours, 'hour');
}

const FINISHED = new Set(['done', 'cached', 'failed']);
/** Bounds on how far the live re-fit may bend the plan: one slow first chapter
 *  (a model load) must not triple the rest of the countdown. */
const MIN_PACE = 0.25;
const MAX_PACE = 4;

/**
 * Seconds left in a chapterized render, re-fitted as chapters finish.
 *
 * `planned` is the estimate's seconds per chapter; `statuses` the tracker's
 * chapter states; `elapsed` seconds since the render started and
 * `sinceLastChapter` seconds since the latest chapter finished. Rendered
 * chapters set the pace (observed / planned); cached ones cost nothing and are
 * left out. The chapter in progress counts down to zero and then waits, so
 * the total never goes below what the chapters not yet started will take.
 * Null when the plan does not describe this render.
 */
export function longformRemaining(
  planned: readonly (number | null)[] | null,
  statuses: readonly string[],
  elapsed: number,
  sinceLastChapter: number,
): number | null {
  if (!planned || planned.length !== statuses.length || planned.length === 0) return null;
  if (planned.some((value) => value === null || !Number.isFinite(value))) return null;
  const plan = planned as number[];
  let plannedDone = 0;
  let rendered = 0;
  statuses.forEach((status, index) => {
    if (status === 'done' || status === 'failed') {
      plannedDone += plan[index];
      rendered += 1;
    }
  });
  const observed = Math.max(0, elapsed - sinceLastChapter);
  const pace =
    rendered > 0 && plannedDone > 0
      ? Math.min(MAX_PACE, Math.max(MIN_PACE, observed / plannedDone))
      : 1;
  const open = statuses
    .map((status, index) => (FINISHED.has(status) ? -1 : index))
    .filter((i) => i >= 0);
  if (open.length === 0) return 0;
  const [current, ...later] = open;
  const currentLeft = Math.max(0, pace * plan[current] - sinceLastChapter);
  return currentLeft + later.reduce((sum, index) => sum + pace * plan[index], 0);
}

/**
 * Fallback countdown with no estimate for this render (cold start, resume):
 * average time per chapter actually rendered, times the chapters left. Cached
 * chapters took no time and are not averaged in. Null until one has rendered.
 */
export function renderedChapterEta(statuses: readonly string[], elapsed: number): number | null {
  const rendered = statuses.filter((status) => status === 'done' || status === 'failed').length;
  const left = statuses.filter((status) => !FINISHED.has(status)).length;
  if (rendered === 0 || left === 0) return null;
  return (elapsed / rendered) * left;
}

/**
 * The backend's live countdown for a running longform render: seconds left
 * after the latest finished call (`remaining`), the part of it that is the
 * call now in flight (`next`), and when it arrived (`at`, performance.now()).
 */
export interface LiveCountdown {
  remaining: number;
  next: number;
  at: number;
}

/**
 * Seconds left now: the in-flight call counts down from its paced cost, but
 * the total never drops below the calls that have not started yet.
 */
export function liveRemaining(live: LiveCountdown, nowMs: number): number {
  const since = Math.max(0, (nowMs - live.at) / 1000);
  const later = Math.max(0, live.remaining - live.next);
  return later + Math.max(0, live.next - since);
}

/** A backend `progress` frame, validated; null when it is not one. */
export function parseLiveCountdown(
  event: Record<string, unknown>,
  nowMs: number,
): LiveCountdown | null {
  const remaining = Number(event.remaining_s);
  const next = Number(event.next_call_s);
  if (event.type !== 'progress' || !Number.isFinite(remaining) || remaining < 0) return null;
  return {
    remaining,
    next: Number.isFinite(next) && next >= 0 ? Math.min(next, remaining) : 0,
    at: nowMs,
  };
}

/** Seconds left of a single take planned at `planned` seconds, never below zero. */
export function takeRemaining(planned: number | null, elapsed: number): number | null {
  if (planned === null || !Number.isFinite(planned)) return null;
  return Math.max(0, planned - Math.max(0, elapsed));
}
