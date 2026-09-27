import { describe, expect, it } from 'vitest';
import {
  formatRenderDuration,
  longformRemaining,
  renderedChapterEta,
  roundEstimateSeconds,
  takeRemaining,
} from './render-estimate-time';

describe('roundEstimateSeconds', () => {
  it('claims only the precision an estimate has', () => {
    expect(roundEstimateSeconds(0)).toBe(0);
    expect(roundEstimateSeconds(2)).toBe(5);
    expect(roundEstimateSeconds(43)).toBe(45);
    expect(roundEstimateSeconds(61)).toBe(60);
    expect(roundEstimateSeconds(276)).toBe(300);
    expect(roundEstimateSeconds(3 * 3600 + 7 * 60)).toBe(3 * 3600 + 5 * 60);
    expect(roundEstimateSeconds(42 * 3600 + 20 * 60)).toBe(42 * 3600);
    expect(roundEstimateSeconds(Number.NaN)).toBe(0);
  });
});

describe('formatRenderDuration', () => {
  it('formats with Intl in the UI locale', () => {
    expect(formatRenderDuration(40, 'en')).toBe('40 sec');
    expect(formatRenderDuration(276, 'en')).toBe('5 min');
    expect(formatRenderDuration(5100, 'en')).toBe('1 hr 25 min');
    expect(formatRenderDuration(7200, 'en')).toBe('2 hr');
    expect(formatRenderDuration(276, 'de')).toBe('5 Min.');
  });
});

describe('longformRemaining', () => {
  const plan = [100, 200, 300];

  it('counts the plan down before any chapter finishes', () => {
    expect(longformRemaining(plan, ['rendering', 'pending', 'pending'], 30, 30)).toBe(570);
  });

  it('re-fits the pace from the chapters actually rendered', () => {
    // Chapter 1 took 200 s against a 100 s plan: this machine is twice as slow today.
    expect(longformRemaining(plan, ['done', 'rendering', 'pending'], 200, 0)).toBe(1000);
  });

  it('leaves cached chapters out of the pace: they cost nothing', () => {
    expect(longformRemaining(plan, ['cached', 'rendering', 'pending'], 1, 0)).toBe(500);
  });

  it('holds at the chapters not yet started when the current one runs long', () => {
    expect(longformRemaining(plan, ['rendering', 'pending', 'pending'], 400, 400)).toBe(500);
    expect(longformRemaining(plan, ['done', 'done', 'rendering'], 300, 900)).toBe(0);
  });

  it('bounds how far one slow chapter bends the plan', () => {
    expect(longformRemaining(plan, ['done', 'pending', 'pending'], 10_000, 0)).toBe(4 * 500);
  });

  it('is zero once every chapter finished, and null for a plan of another render', () => {
    expect(longformRemaining(plan, ['done', 'cached', 'failed'], 50, 0)).toBe(0);
    expect(longformRemaining(plan, ['pending', 'pending'], 0, 0)).toBeNull();
    expect(longformRemaining(null, ['pending'], 0, 0)).toBeNull();
    expect(longformRemaining([100, null], ['pending', 'pending'], 0, 0)).toBeNull();
  });
});

describe('fallback countdowns', () => {
  it('averages rendered chapters only', () => {
    expect(renderedChapterEta(['cached', 'done', 'pending', 'pending'], 60)).toBe(120);
    expect(renderedChapterEta(['cached', 'rendering'], 5)).toBeNull();
    expect(renderedChapterEta(['done', 'done'], 5)).toBeNull();
  });

  it('never counts a take below zero', () => {
    expect(takeRemaining(120, 20)).toBe(100);
    expect(takeRemaining(120, 500)).toBe(0);
    expect(takeRemaining(null, 5)).toBeNull();
  });
});
