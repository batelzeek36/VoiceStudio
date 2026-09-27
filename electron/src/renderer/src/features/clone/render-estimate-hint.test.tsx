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
    'Estimate appears after your first render on this machine',
  );
  expect(estimateLine({ ...cold, reason: 'remote' }, i18next.t, 'en')).toBeNull();
  expect(estimateLine(null, i18next.t, 'en')).toBeNull();
});
