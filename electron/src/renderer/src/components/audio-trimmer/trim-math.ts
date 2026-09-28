/**
 * Pure selection math for the audio trimmer: length presets, fit grading,
 * silence snapping and the default window. Kept free of DOM and wavesurfer so
 * every rule the UI relies on is unit-tested rather than eyeballed.
 */

export interface TrimRange {
  start: number;
  end: number;
}

/**
 * Which edge(s) of a selection just moved, so snapping only touches those:
 * `start`/`end` one edge, `move` the whole selection (length kept), `edges` a
 * freshly drawn selection (each edge on its own), `both` a length preset (start
 * first, then the length from there).
 */
export type TrimSide = 'start' | 'end' | 'move' | 'edges' | 'both';

/** Nothing shorter than this is exported; the engines have no use for it. */
export const TRIM_MIN_SECONDS = 0.5;
/** Below this the readout warns: too little speech to learn a voice from. */
export const TRIM_SHORT_SECONDS = 3;
/** The range that clones best (clone.reference_hint says 5–15; 8–15 is the sweet spot). */
export const TRIM_BEST_MIN_SECONDS = 8;
export const TRIM_BEST_MAX_SECONDS = 15;
/** A snapped cut may move this far to land in a pause. */
export const SNAP_RADIUS_SECONDS = 0.3;
/** Energy is measured per hop; a cut lands on a hop boundary. */
export const ENVELOPE_HOP_SECONDS = 0.01;
/** Speech starts where the level first rises above this share of the loudest hop. */
export const SPEECH_ONSET_RATIO = 0.1;
/** Lead-in kept before the first detected speech in the default window. */
export const SPEECH_ONSET_LEAD_SECONDS = 0.1;

export type TrimFit = 'short' | 'best' | 'ok' | 'long';

const EPSILON = 1e-6;

const clamp = (value: number, low: number, high: number): number =>
  Math.max(low, Math.min(high, value));

export function trimFit(lengthSeconds: number, maxSeconds: number): TrimFit {
  if (lengthSeconds > maxSeconds + EPSILON) return 'long';
  if (lengthSeconds < TRIM_SHORT_SECONDS) return 'short';
  if (lengthSeconds >= TRIM_BEST_MIN_SECONDS && lengthSeconds <= TRIM_BEST_MAX_SECONDS + EPSILON)
    return 'best';
  return 'ok';
}

/**
 * Keep a range inside the clip and inside [minSeconds, maxSeconds]. The start
 * wins over the end: an over-long range is shortened from the end, an
 * under-short one is grown from the end (and only pulled back at the clip end).
 */
export function clampRange(
  range: TrimRange,
  duration: number,
  maxSeconds: number,
  minSeconds = TRIM_MIN_SECONDS,
): TrimRange {
  const length = Math.max(0, duration);
  const floor = Math.min(minSeconds, length);
  let start = clamp(Number.isFinite(range.start) ? range.start : 0, 0, length);
  let end = clamp(Number.isFinite(range.end) ? range.end : length, start, length);
  if (end - start > maxSeconds) end = start + maxSeconds;
  if (end - start < floor) {
    end = Math.min(length, start + floor);
    if (end - start < floor) start = Math.max(0, end - floor);
  }
  return { start, end };
}

/**
 * A length preset keeps the selection's start and sets its length; near the
 * end of the clip the window slides back so it stays whole, and it never
 * reaches past the clip.
 */
export function presetRange(start: number, lengthSeconds: number, duration: number): TrimRange {
  const length = Math.max(0, duration);
  const safeStart = clamp(
    Number.isFinite(start) ? start : 0,
    0,
    Math.max(0, length - lengthSeconds),
  );
  return { start: safeStart, end: Math.min(length, safeStart + lengthSeconds) };
}

/** Per-hop mean-square energy of a mono buffer; `rms(i)` reads a 2-hop (20 ms) window. */
export interface Envelope {
  hopSeconds: number;
  duration: number;
  /** Mean square per hop; length = ceil(duration / hop). */
  power: Float32Array;
}

export function rmsEnvelope(
  samples: ArrayLike<number>,
  sampleRate: number,
  hopSeconds = ENVELOPE_HOP_SECONDS,
): Envelope {
  const hopSamples = Math.max(1, Math.round(hopSeconds * sampleRate));
  const hops = Math.max(1, Math.ceil(samples.length / hopSamples));
  const power = new Float32Array(hops);
  for (let hop = 0; hop < hops; hop++) {
    const from = hop * hopSamples;
    const to = Math.min(samples.length, from + hopSamples);
    let sum = 0;
    for (let index = from; index < to; index++) sum += samples[index] * samples[index];
    power[hop] = to > from ? sum / (to - from) : 0;
  }
  return { hopSeconds: hopSamples / sampleRate, duration: samples.length / sampleRate, power };
}

/** RMS of the 20 ms window centred on hop boundary `index` (time = index * hop). */
export function boundaryRms(envelope: Envelope, index: number): number {
  const { power } = envelope;
  const before = index - 1 >= 0 && index - 1 < power.length ? power[index - 1] : null;
  const after = index >= 0 && index < power.length ? power[index] : null;
  if (before === null && after === null) return 0;
  if (before === null) return Math.sqrt(after ?? 0);
  if (after === null) return Math.sqrt(before);
  return Math.sqrt((before + after) / 2);
}

/**
 * The quietest hop boundary within `radius` of `time`, limited to [low, high].
 * Ties go to the boundary nearest the requested time, so snapping never drifts
 * across a long silence. Returns the clamped time itself when nothing fits.
 */
export function quietestNear(
  envelope: Envelope,
  time: number,
  {
    before = SNAP_RADIUS_SECONDS,
    after = SNAP_RADIUS_SECONDS,
    low = 0,
    high = envelope.duration,
  } = {},
): number {
  const floor = Math.max(low, time - before);
  const ceiling = Math.min(high, time + after);
  if (ceiling < floor) return clamp(time, low, high);
  const hop = envelope.hopSeconds;
  const first = Math.ceil(floor / hop - EPSILON);
  const last = Math.floor(ceiling / hop + EPSILON);
  if (last < first) return clamp(time, floor, ceiling);
  let bestIndex = first;
  let bestRms = Number.POSITIVE_INFINITY;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = first; index <= last; index++) {
    const rms = boundaryRms(envelope, index);
    const distance = Math.abs(index * hop - time);
    if (
      rms < bestRms - EPSILON ||
      (Math.abs(rms - bestRms) <= EPSILON && distance < bestDistance)
    ) {
      bestIndex = index;
      bestRms = rms;
      bestDistance = distance;
    }
  }
  return clamp(bestIndex * hop, floor, ceiling);
}

/**
 * Snap the edge(s) that moved to the nearest pause. `start`/`end` snap that
 * edge within ±radius; `edges` snaps each edge of a freshly drawn selection on
 * its own; `move` snaps the start and keeps the length, so a dragged preset
 * stays the length the user chose; `both` (a preset or the default window)
 * snaps the start either way, measures the length from there and lets the end
 * move earlier only, so a preset never grows past the length it was given. An
 * edge sitting on the clip boundary stays there. The result always respects
 * the clip, `minSeconds` and `maxSeconds`.
 */
export function snapRange(
  envelope: Envelope,
  range: TrimRange,
  side: TrimSide,
  { maxSeconds, minSeconds = TRIM_MIN_SECONDS }: { maxSeconds: number; minSeconds?: number },
): TrimRange {
  const duration = envelope.duration;
  const base = clampRange(range, duration, maxSeconds, minSeconds);
  const floor = Math.min(minSeconds, duration);
  const snapStart = (start: number, end: number): number =>
    start <= EPSILON
      ? 0
      : quietestNear(envelope, start, {
          low: Math.max(0, end - maxSeconds),
          high: Math.max(0, Math.min(end - floor, duration - floor)),
        });
  const snapEnd = (start: number, end: number, after = SNAP_RADIUS_SECONDS): number =>
    end >= duration - EPSILON
      ? end
      : quietestNear(envelope, end, {
          after,
          low: start + floor,
          high: Math.min(duration, start + maxSeconds),
        });
  const settle = (next: TrimRange): TrimRange => clampRange(next, duration, maxSeconds, minSeconds);
  if (side === 'start') return settle({ start: snapStart(base.start, base.end), end: base.end });
  if (side === 'end') return settle({ start: base.start, end: snapEnd(base.start, base.end) });
  if (side === 'edges') {
    const start = snapStart(base.start, base.end);
    return settle({ start, end: snapEnd(start, base.end) });
  }
  const length = base.end - base.start;
  const start =
    base.start <= EPSILON
      ? 0
      : quietestNear(envelope, base.start, { high: Math.max(0, duration - floor) });
  const end = Math.min(duration, start + length);
  if (side === 'move') return settle({ start, end });
  return settle({ start, end: snapEnd(start, end, 0) });
}

/** Where speech first rises above SPEECH_ONSET_RATIO of the loudest hop, minus a short lead-in. */
export function speechOnset(envelope: Envelope): number {
  let peak = 0;
  for (const value of envelope.power) if (value > peak) peak = value;
  if (peak <= 0) return 0;
  const threshold = peak * SPEECH_ONSET_RATIO * SPEECH_ONSET_RATIO;
  for (let hop = 0; hop < envelope.power.length; hop++)
    if (envelope.power[hop] >= threshold)
      return Math.max(0, hop * envelope.hopSeconds - SPEECH_ONSET_LEAD_SECONDS);
  return 0;
}

/**
 * The window the trimmer opens with: `lengthSeconds` from the first speech,
 * slid back near the end of the clip, snapped into pauses when asked.
 */
export function defaultWindow(
  envelope: Envelope,
  lengthSeconds: number,
  {
    maxSeconds,
    snap = true,
    minSeconds = TRIM_MIN_SECONDS,
  }: {
    maxSeconds: number;
    snap?: boolean;
    minSeconds?: number;
  },
): TrimRange {
  const range = presetRange(speechOnset(envelope), lengthSeconds, envelope.duration);
  return snap
    ? snapRange(envelope, range, 'both', { maxSeconds, minSeconds })
    : clampRange(range, envelope.duration, maxSeconds, minSeconds);
}
