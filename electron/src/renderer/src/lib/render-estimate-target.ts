import type { TFunction } from 'i18next';
import type { RenderEstimate } from '@/lib/api/render-estimate';
import { formatRenderDuration } from '@/lib/render-estimate-time';

/**
 * Where an estimate's render runs, in words (docs/adr/render-time-estimate.md):
 * shared by the Voice cloning hint and the longform stats line. Pure: the
 * caller passes its `t`.
 */

/** The worker the render goes to, or null when it runs on this machine. */
export function remoteWorker(estimate: RenderEstimate): string | null {
  return estimate.target.kind === 'local' ? null : estimate.target.label;
}

/** "Usually 3 min to 4 min on this machine" / "... on KAIROS". */
export function estimateRange(
  estimate: RenderEstimate,
  t: TFunction,
  locale: string,
): string | undefined {
  if (estimate.low === null || estimate.high === null) return undefined;
  const low = formatRenderDuration(estimate.low, locale);
  const high = formatRenderDuration(estimate.high, locale);
  const worker = remoteWorker(estimate);
  return worker
    ? t('renderEstimate.range_on', { low, high, worker })
    : t('renderEstimate.range', { low, high });
}

/** "KAIROS is offline" when the chosen worker cannot take the render. */
export function unavailableLine(estimate: RenderEstimate, t: TFunction): string | null {
  if (estimate.target.kind !== 'unavailable') return null;
  return t(estimate.target.offline ? 'renderEstimate.offline' : 'renderEstimate.unavailable', {
    worker: estimate.target.label,
  });
}
