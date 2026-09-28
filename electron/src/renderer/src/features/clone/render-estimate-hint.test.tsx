import { expect, it } from 'vitest';
import i18next from 'i18next';
import '@/i18n';
import type { RenderEstimate } from '@/lib/api/render-estimate';
import { estimateLine } from './render-estimate-hint';

const measured: RenderEstimate = {
  basis: 'measured',
  reason: null,
  seconds: 276,
  low: 250,
  high: 320,
  calls: 1,
  samples: 5,
  parts: [{ calls: 1, seconds: 276 }],
  target: { kind: 'local' },
};

it('says how long the take will render beside Synthesize', () => {
  expect(estimateLine(measured, i18next.t, 'en')).toEqual({
    text: 'About 5 min to render',
    title: 'Usually 4 min to 5 min on this machine',
  });
});

it('marks rough estimates and explains the calibration render', () => {
  expect(estimateLine({ ...measured, basis: 'rough' }, i18next.t, 'en')?.text).toBe(
    'Roughly 5 min to render',
  );
  const cold = {
    ...measured,
    basis: 'none' as const,
    reason: 'cold_start' as const,
    seconds: null,
  };
  expect(estimateLine(cold, i18next.t, 'en')?.text).toBe(
    'Estimate appears after a few renders on this machine',
  );
  expect(estimateLine({ ...cold, reason: 'no_rate' }, i18next.t, 'en')).toBeNull();
  expect(estimateLine(null, i18next.t, 'en')).toBeNull();
});

it('prices a remote render on its worker and says where', () => {
  const remote: RenderEstimate = { ...measured, target: { kind: 'remote', label: 'KAIROS' } };
  expect(estimateLine(remote, i18next.t, 'en')).toEqual({
    text: 'About 5 min to render',
    title: 'Usually 4 min to 5 min on KAIROS',
  });
  expect(
    estimateLine({ ...remote, basis: 'none', reason: 'cold_start', seconds: null }, i18next.t, 'en')
      ?.text,
  ).toBe('Estimate appears after a few renders on KAIROS');
});

it('says the chosen worker is offline instead of estimating', () => {
  const offline: RenderEstimate = {
    ...measured,
    basis: 'none',
    reason: 'remote_unavailable',
    seconds: null,
    target: { kind: 'unavailable', label: 'KAIROS', offline: true },
  };
  expect(estimateLine(offline, i18next.t, 'en')).toEqual({
    text: 'KAIROS is offline',
    unavailable: true,
  });
  expect(
    estimateLine(
      { ...offline, target: { kind: 'unavailable', label: 'KAIROS', offline: false } },
      i18next.t,
      'en',
    )?.text,
  ).toBe("KAIROS can't take renders right now");
});
