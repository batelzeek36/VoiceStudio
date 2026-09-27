import type { RenderEstimate, RenderEstimateRequest } from '@/lib/api/render-estimate';
import { renderBody, type Draft, type Mode } from './longform-session';

/** Output packaging, not synthesis: they never change how long a render takes. */
const OUTPUT_ONLY = new Set(['metadata', 'format', 'loudness', 'cover_path', 'bitrate']);

/**
 * The render-time estimate request for this draft: the synthesis inputs of the
 * exact body `renderLongform` posts, so the estimate plans the same chapters,
 * voices, markup and overrides. Title or cover edits do not re-ask.
 */
export function longformEstimateRequest(mode: Mode, draft: Draft): RenderEstimateRequest {
  const body = renderBody(mode, draft) as Record<string, unknown>;
  const synthesis = Object.fromEntries(
    Object.entries(body).filter(([key]) => !OUTPUT_ONLY.has(key)),
  );
  return { ...synthesis, surface: mode === 'audiobook' ? 'audiobook' : 'longform' };
}

/** Planned seconds per chapter, or null when the estimate has no number. */
export function plannedChapterSeconds(estimate: RenderEstimate | null): (number | null)[] | null {
  if (!estimate || estimate.seconds === null || estimate.parts.length === 0) return null;
  return estimate.parts.map((part) => part.seconds);
}
