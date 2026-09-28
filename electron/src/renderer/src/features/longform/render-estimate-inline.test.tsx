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
  target: { kind: 'local' },
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

it('names the calibration render on a fresh machine, and stays quiet without a reason', () => {
  const cold = { ...base, basis: 'none' as const, reason: 'cold_start' as const, seconds: null };
  const { rerender, container } = render(<RenderEstimateInline estimate={cold} />);
  expect(screen.getByText(/after a few renders on this machine/)).toBeVisible();
  rerender(<RenderEstimateInline estimate={{ ...cold, reason: 'no_rate' }} />);
  expect(container).toBeEmptyDOMElement();
});

it('prices a remote render on its worker, or says the worker is offline', () => {
  const remote: RenderEstimate = { ...base, target: { kind: 'remote', label: 'KAIROS' } };
  const { rerender } = render(<RenderEstimateInline estimate={remote} />);
  expect(screen.getByText(/render about 13 min/)).toHaveAttribute(
    'title',
    'Usually 12 min to 15 min on KAIROS',
  );
  rerender(
    <RenderEstimateInline
      estimate={{ ...remote, basis: 'none', reason: 'cold_start', seconds: null }}
    />,
  );
  expect(screen.getByText(/after a few renders on KAIROS/)).toBeVisible();
  rerender(
    <RenderEstimateInline
      estimate={{
        ...remote,
        basis: 'none',
        reason: 'remote_unavailable',
        seconds: null,
        target: { kind: 'unavailable', label: 'KAIROS', offline: true },
      }}
    />,
  );
  expect(screen.getByText(/KAIROS is offline/)).toBeVisible();
});
