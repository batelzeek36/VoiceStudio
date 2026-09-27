import { expect, it } from 'vitest';
import { blankLongformDraft } from './longform-session';
import { longformEstimateRequest, plannedChapterSeconds } from './longform-estimate';

it('asks with the synthesis inputs of the render body, not its packaging', () => {
  const draft = {
    ...blankLongformDraft(),
    script: '# One\nHello.',
    voice: 'narrator',
    title: 'My book',
    format: 'mp3' as const,
  };
  const request = longformEstimateRequest('audiobook', draft);
  expect(request).toMatchObject({
    surface: 'audiobook',
    text: '# One\nHello.',
    default_voice: 'narrator',
  });
  for (const key of ['metadata', 'format', 'loudness', 'cover_path'])
    expect(request).not.toHaveProperty(key);
  // A title edit does not change the request, so it never re-asks.
  expect(JSON.stringify(longformEstimateRequest('audiobook', { ...draft, title: 'Renamed' }))).toBe(
    JSON.stringify(request),
  );
});

it('sends Stories as compiled chapters', () => {
  const draft = {
    ...blankLongformDraft(),
    voice: 'narrator',
    lines: [{ id: 'a', text: 'A line.', profileId: null }],
  };
  const request = longformEstimateRequest('stories', draft);
  expect(request.surface).toBe('longform');
  expect(Array.isArray(request.chapters)).toBe(true);
});

it('plans chapters only when the estimate has a number', () => {
  const parts = [
    { calls: 2, seconds: 30 },
    { calls: 1, seconds: 10 },
  ];
  const base = { reason: null, low: 30, high: 50, calls: 3, samples: 5, parts };
  expect(plannedChapterSeconds({ ...base, basis: 'measured', seconds: 40 })).toEqual([30, 10]);
  expect(
    plannedChapterSeconds({ ...base, basis: 'none', reason: 'cold_start', seconds: null }),
  ).toBeNull();
  expect(plannedChapterSeconds(null)).toBeNull();
});
