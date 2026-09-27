import { expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import '@/i18n';
import { GenerationProgress } from './generation-progress';

const chapters = (...statuses: string[]) => statuses.map((status) => ({ title: status, status }));

it('counts down from the measured plan when a render starts', () => {
  render(
    <GenerationProgress
      chapters={chapters('rendering', 'pending', 'pending')}
      assembling={false}
      planned={[100, 200, 300]}
    />,
  );
  expect(screen.getByText(/10:00 left/)).toBeVisible();
});

it('says it is finishing up instead of going below zero', () => {
  render(
    <GenerationProgress
      chapters={chapters('done', 'cached', 'done')}
      assembling={false}
      planned={[100, 200, 300]}
    />,
  );
  expect(screen.getByText(/Finishing up/)).toBeVisible();
  expect(screen.queryByText(/left/)).toBeNull();
});

it('shows no countdown before anything rendered when there is no plan', () => {
  render(<GenerationProgress chapters={chapters('rendering', 'pending')} assembling={false} />);
  expect(screen.queryByText(/left|Finishing up/)).toBeNull();
});

it('ignores a plan that describes a different render', () => {
  render(
    <GenerationProgress
      chapters={chapters('rendering', 'pending')}
      assembling={false}
      planned={[100, 200, 300]}
    />,
  );
  expect(screen.queryByText(/left|Finishing up/)).toBeNull();
});
