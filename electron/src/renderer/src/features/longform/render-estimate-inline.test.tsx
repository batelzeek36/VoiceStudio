import { expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import '@/i18n';
import type { RenderEstimate } from '@/lib/api/render-estimate';
import { RenderEstimateInline } from './render-estimate-inline';

const base: RenderEstimate = {
  basis: 'measured',
  reason: null,
  seconds: 780,
  low: 700,
  high: 900,
  calls: 40,
  samples: 12,
  parts: [{ calls: 40, seconds: 780 }],
};

it('reads "render about 13 min" after the runtime estimate', () => {
  render(<RenderEstimateInline estimate={base} />);
  const text = screen.getByText(/render about 13 min/);
  expect(text).toHaveAttribute('title', 'Usually 12 min to 15 min on this machine');
});

it('marks a steps-scaled estimate as rough and says why', () => {
  render(<RenderEstimateInline estimate={{ ...base, basis: 'rough' }} />);
  expect(screen.getByText(/render roughly 13 min/).getAttribute('title')).toMatch(
    /other quality settings/,
  );
});

it('names the calibration render on a fresh machine, and stays quiet for remote renders', () => {
  const cold = { ...base, basis: 'none' as const, reason: 'cold_start' as const, seconds: null };
  const { rerender, container } = render(<RenderEstimateInline estimate={cold} />);
  expect(screen.getByText(/after a few renders on this machine/)).toBeVisible();
  rerender(<RenderEstimateInline estimate={{ ...cold, reason: 'remote' }} />);
  expect(container).toBeEmptyDOMElement();
});
