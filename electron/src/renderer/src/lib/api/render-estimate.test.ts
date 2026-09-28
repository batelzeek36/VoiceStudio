import { describe, expect, it } from 'vitest';
import { DEFAULT_CLONE_SETTINGS } from '@/lib/store/clone-settings';
import { cloneEstimateRequest, parseRenderEstimate } from './render-estimate';

const measured = {
  basis: 'measured',
  reason: null,
  seconds: 812.4,
  low: 700,
  high: 950,
  calls: 18,
  samples: 42,
  parts: [
    { calls: 10, seconds: 500.2 },
    { calls: 8, seconds: 312.2 },
  ],
};

describe('parseRenderEstimate', () => {
  it('accepts a measured estimate', () => {
    expect(parseRenderEstimate(measured)).toEqual({
      basis: 'measured',
      reason: null,
      seconds: 812.4,
      low: 700,
      high: 950,
      calls: 18,
      samples: 42,
      parts: [
        { calls: 10, seconds: 500.2 },
        { calls: 8, seconds: 312.2 },
      ],
      target: { kind: 'local' },
    });
  });

  it('names the remote worker the render goes to', () => {
    const remote = parseRenderEstimate({
      ...measured,
      target: { kind: 'remote', label: 'KAIROS' },
    });
    expect(remote).toMatchObject({ seconds: 812.4, target: { kind: 'remote', label: 'KAIROS' } });
  });

  it('never prices a render for a worker that cannot take it', () => {
    const offline = parseRenderEstimate({
      ...measured,
      basis: 'none',
      reason: 'remote_unavailable',
      seconds: null,
      target: { kind: 'unavailable', label: 'KAIROS', offline: true },
    });
    expect(offline).toMatchObject({
      basis: 'none',
      reason: 'remote_unavailable',
      seconds: null,
      target: { kind: 'unavailable', label: 'KAIROS', offline: true },
    });
    // Even a (buggy) number next to an unavailable target is not shown.
    const contradictory = parseRenderEstimate({
      ...measured,
      target: { kind: 'unavailable', label: 'KAIROS' },
    });
    expect(contradictory).toMatchObject({ seconds: null, reason: 'remote_unavailable' });
  });

  it('shows nothing when the target cannot be read', () => {
    expect(parseRenderEstimate({ ...measured, target: { kind: 'remote' } })).toBeNull();
    expect(parseRenderEstimate({ ...measured, target: { kind: 'mars', label: 'x' } })).toBeNull();
    expect(
      parseRenderEstimate({ ...measured, target: { kind: 'remote', label: 'x'.repeat(65) } }),
    ).toBeNull();
    expect(parseRenderEstimate({ ...measured, target: 'KAIROS' })).toBeNull();
  });

  it('keeps the cold-start reason and never invents a number', () => {
    const cold = parseRenderEstimate({
      ...measured,
      basis: 'none',
      reason: 'cold_start',
      seconds: null,
    });
    expect(cold).toMatchObject({ basis: 'none', reason: 'cold_start', seconds: null, low: null });
    expect(cold?.parts.every((part) => part.seconds === null)).toBe(true);
  });

  it('degrades malformed payloads to no number', () => {
    expect(parseRenderEstimate(null)).toBeNull();
    expect(parseRenderEstimate([])).toBeNull();
    expect(parseRenderEstimate({ ...measured, seconds: 'soon' })).toMatchObject({
      basis: 'none',
      seconds: null,
    });
    expect(parseRenderEstimate({ ...measured, basis: 'guess' })?.basis).toBe('none');
    expect(parseRenderEstimate({ ...measured, seconds: Number.NaN })?.seconds).toBeNull();
    expect(parseRenderEstimate({ ...measured, parts: 'x' })?.parts).toEqual([]);
  });

  it('falls back to the total when the range is missing', () => {
    const noRange = parseRenderEstimate({ ...measured, low: null, high: undefined });
    expect(noRange).toMatchObject({ low: 812.4, high: 812.4 });
  });
});

describe('cloneEstimateRequest', () => {
  const settings = { ...DEFAULT_CLONE_SETTINGS, text: 'Hello there.', steps: 32, speed: 1.2 };

  it('asks nothing for an empty script', () => {
    expect(cloneEstimateRequest({ ...settings, text: '   ' }, null)).toBeNull();
  });

  it('sends what /generate receives for a saved voice', () => {
    expect(
      cloneEstimateRequest({ ...settings, selectedProfileId: 'p1', refText: 'stored' }, 9),
    ).toMatchObject({
      surface: 'generate',
      profile_id: 'p1',
      ref_text: null,
      ref_seconds: null,
      num_step: 32,
      speed: 1.2,
      duration: null,
    });
  });

  it('describes an uploaded reference by its transcript and length', () => {
    expect(cloneEstimateRequest({ ...settings, refText: 'A clip.' }, 6.5)).toMatchObject({
      profile_id: null,
      ref_text: 'A clip.',
      ref_seconds: 6.5,
    });
  });

  it('passes only a usable explicit duration and a validator-safe instruct', () => {
    expect(cloneEstimateRequest({ ...settings, duration: '12.5' }, null)?.duration).toBe(12.5);
    expect(cloneEstimateRequest({ ...settings, duration: 'abc' }, null)?.duration).toBeNull();
    expect(cloneEstimateRequest({ ...settings, duration: '99999' }, null)?.duration).toBeNull();
    expect(
      cloneEstimateRequest({ ...settings, instruct: 'female, a warm narrator' }, null)?.instruct,
    ).toBe('female');
  });
});
