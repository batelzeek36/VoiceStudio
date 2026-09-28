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

it('follows the backend countdown between chapter events', () => {
  render(
    <GenerationProgress
      chapters={chapters('rendering')}
      assembling={false}
      planned={[600]}
      live={{ remaining: 125, next: 25, at: performance.now() }}
    />,
  );
  // The per-call countdown (about 2:05), not the one-chapter plan (10:00).
  expect(screen.getByText(/2:0[45] left/)).toBeVisible();
});

it('finishes up when the backend has nothing left for the last chapter', () => {
  render(
    <GenerationProgress
      chapters={chapters('rendering')}
      assembling={false}
      live={{ remaining: 0, next: 0, at: performance.now() }}
    />,
  );
  expect(screen.getByText(/Finishing up/)).toBeVisible();
});
