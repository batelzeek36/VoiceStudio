import { describe, expect, it } from 'vitest';
import {
  clampRange,
  defaultWindow,
  presetRange,
  quietestNear,
  rmsEnvelope,
  snapRange,
  speechOnset,
  trimFit,
} from './trim-math';

/** A 1 kHz mono signal: loud (0.5) everywhere except the given silent spans. */
function signal(seconds: number, silent: Array<[number, number]>, sampleRate = 1000): Float32Array {
  const samples = new Float32Array(Math.round(seconds * sampleRate)).fill(0.5);
  for (const [from, to] of silent)
    samples.fill(0, Math.round(from * sampleRate), Math.round(to * sampleRate));
  return samples;
}

describe('presetRange', () => {
  it('keeps the start and sets the length', () => {
    expect(presetRange(4.2, 10, 35.8)).toEqual({ start: 4.2, end: 14.2 });
    expect(presetRange(4.2, 20, 35.8)).toEqual({ start: 4.2, end: 24.2 });
  });
  it('slides back near the end of the clip instead of running past it', () => {
    const late = presetRange(30, 10, 35.8);
    expect(late.start).toBeCloseTo(25.8, 9);
    expect(late.end).toBeCloseTo(35.8, 9);
    const last = presetRange(35.8, 15, 35.8);
    expect(last.start).toBeCloseTo(20.8, 9);
    expect(last.end).toBeCloseTo(35.8, 9);
  });
  it('never reaches past a clip shorter than the preset', () => {
    expect(presetRange(3, 20, 12)).toEqual({ start: 0, end: 12 });
    expect(presetRange(NaN, 10, 12)).toEqual({ start: 0, end: 10 });
  });
});

describe('clampRange', () => {
  it('shortens an over-long selection from the end and keeps it inside the clip', () => {
    expect(clampRange({ start: 2, end: 30 }, 35.8, 20)).toEqual({ start: 2, end: 22 });
    expect(clampRange({ start: -1, end: 99 }, 12, 20)).toEqual({ start: 0, end: 12 });
  });
  it('grows a selection shorter than the floor, pulling back only at the clip end', () => {
    expect(clampRange({ start: 5, end: 5.1 }, 35.8, 20)).toEqual({ start: 5, end: 5.5 });
    expect(clampRange({ start: 35.7, end: 35.8 }, 35.8, 20)).toEqual({ start: 35.3, end: 35.8 });
  });
});

describe('trimFit', () => {
  it('grades the length against the recommended range and the cap', () => {
    expect(trimFit(2.9, 20)).toBe('short');
    expect(trimFit(5, 20)).toBe('ok');
    expect(trimFit(8, 20)).toBe('best');
    expect(trimFit(15, 20)).toBe('best');
    expect(trimFit(17, 20)).toBe('ok');
    expect(trimFit(20, 20)).toBe('ok');
    expect(trimFit(20.01, 20)).toBe('long');
  });
});

describe('snapping', () => {
  const envelope = rmsEnvelope(
    signal(30, [
      [1.2, 1.3],
      [10.5, 10.6],
      [19.7, 19.75],
    ]),
    1000,
  );

  it('measures energy per 10 ms hop', () => {
    expect(envelope.hopSeconds).toBeCloseTo(0.01, 9);
    expect(envelope.duration).toBe(30);
    expect(envelope.power[0]).toBeCloseTo(0.25, 6);
    expect(envelope.power[125]).toBe(0);
  });

  it('finds the quietest boundary within the radius, nearest first on ties', () => {
    // From 1.0 the radius reaches 1.3; the gap's interior boundaries all read
    // zero, so the nearest of them wins.
    expect(quietestNear(envelope, 1.0)).toBeCloseTo(1.21, 6);
    // From 1.5 the same gap is reached from the other side.
    expect(quietestNear(envelope, 1.5)).toBeCloseTo(1.29, 6);
    // Nothing quiet within reach: stay put.
    expect(quietestNear(envelope, 5)).toBeCloseTo(5, 6);
  });

  it('respects the low/high bounds and the one-sided radius', () => {
    expect(quietestNear(envelope, 1.0, { low: 1.25 })).toBeCloseTo(1.25, 6);
    expect(quietestNear(envelope, 1.0, { after: 0 })).toBeCloseTo(1.0, 6);
    expect(quietestNear(envelope, 1.5, { high: 1.0 })).toBeCloseTo(1.0, 6);
  });

  it('snaps only the edge that moved', () => {
    const start = snapRange(envelope, { start: 1.0, end: 8 }, 'start', { maxSeconds: 20 });
    expect(start.start).toBeCloseTo(1.21, 6);
    expect(start.end).toBe(8);
    const end = snapRange(envelope, { start: 4, end: 10.4 }, 'end', { maxSeconds: 20 });
    expect(end.start).toBe(4);
    expect(end.end).toBeCloseTo(10.51, 6);
  });

  it('keeps the length when the whole selection is moved', () => {
    const moved = snapRange(envelope, { start: 1.0, end: 11.0 }, 'move', { maxSeconds: 20 });
    expect(moved.start).toBeCloseTo(1.21, 6);
    expect(moved.end - moved.start).toBeCloseTo(10, 6);
  });

  it('snaps each edge of a freshly drawn selection on its own', () => {
    const fresh = snapRange(envelope, { start: 1.0, end: 19.9 }, 'edges', { maxSeconds: 20 });
    expect(fresh.start).toBeCloseTo(1.21, 6);
    expect(fresh.end).toBeCloseTo(19.74, 6);
  });

  it('measures a preset from the snapped start and lets its end move earlier only', () => {
    const gapped = rmsEnvelope(
      signal(30, [
        [1.2, 1.3],
        [16.0, 16.1],
      ]),
      1000,
    );
    const preset = snapRange(gapped, { start: 1.0, end: 16.0 }, 'both', { maxSeconds: 20 });
    expect(preset.start).toBeCloseTo(1.21, 6);
    // 1.21 + 15 = 16.21; the gap just before it is the nearest quiet boundary.
    expect(preset.end).toBeCloseTo(16.09, 6);
    expect(preset.end - preset.start).toBeLessThanOrEqual(15);
  });

  it('never lets a snapped selection exceed the cap', () => {
    // A 20 s preset from 1.0 ends at 21.0; the start snaps later, the end may
    // only move earlier, so the result stays within 20 s.
    const range = snapRange(envelope, { start: 1.0, end: 21.0 }, 'both', { maxSeconds: 20 });
    expect(range.end - range.start).toBeLessThanOrEqual(20 + 1e-6);
    const widened = snapRange(envelope, { start: 1.6, end: 21.5 }, 'start', { maxSeconds: 20 });
    expect(widened.end - widened.start).toBeLessThanOrEqual(20 + 1e-6);
  });

  it('leaves an edge that sits on the clip boundary alone', () => {
    const atStart = snapRange(envelope, { start: 0, end: 10 }, 'both', { maxSeconds: 20 });
    expect(atStart.start).toBe(0);
    const atEnd = snapRange(envelope, { start: 20, end: 30 }, 'end', { maxSeconds: 20 });
    expect(atEnd.end).toBe(30);
  });
});

describe('defaultWindow', () => {
  it('starts just before the first speech and takes the requested length', () => {
    const envelope = rmsEnvelope(
      signal(30, [
        [0, 2],
        [16.8, 16.9],
      ]),
      1000,
    );
    expect(speechOnset(envelope)).toBeCloseTo(1.9, 6);
    const window = defaultWindow(envelope, 15, { maxSeconds: 20 });
    // The 0.1 s lead-in sits in the silence, so the start is already quiet.
    expect(window.start).toBeCloseTo(1.9, 6);
    // 1.9 + 15 = 16.9 lands in the pause; its nearest quiet boundary is kept.
    expect(window.end).toBeCloseTo(16.89, 6);
  });
  it('slides back when the speech starts near the end of the clip', () => {
    const envelope = rmsEnvelope(signal(20, [[0, 12]]), 1000);
    const window = defaultWindow(envelope, 15, { maxSeconds: 20, snap: false });
    expect(window).toEqual({ start: 5, end: 20 });
  });
  it('takes the whole clip when it is shorter than the window', () => {
    const envelope = rmsEnvelope(signal(8, []), 1000);
    expect(defaultWindow(envelope, 15, { maxSeconds: 20 })).toEqual({ start: 0, end: 8 });
  });
});
