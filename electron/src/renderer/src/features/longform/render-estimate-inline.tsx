import { useTranslation } from 'react-i18next';
import type { RenderEstimate } from '@/lib/api/render-estimate';
import { estimateRange, remoteWorker, unavailableLine } from '@/lib/render-estimate-target';
import { formatRenderDuration } from '@/lib/render-estimate-time';

/**
 * After the runtime estimate in the script stats line: how long rendering it
 * will take where it runs ("render about 13 min"; the tooltip names a remote
 * worker), when that number will appear, or that the chosen worker cannot
 * take it. Nothing while the first estimate loads.
 */
export function RenderEstimateInline({ estimate }: { estimate: RenderEstimate | null }) {
  const { t, i18n } = useTranslation();
  if (!estimate) return null;
  const unavailable = unavailableLine(estimate, t);
  if (unavailable) return <span data-slot="render-estimate"> · {unavailable}</span>;
  if (estimate.seconds === null) {
    if (estimate.reason !== 'cold_start') return null;
    const worker = remoteWorker(estimate);
    return (
      <span data-slot="render-estimate">
        {' · '}
        {worker ? t('renderEstimate.inline_cold_on', { worker }) : t('renderEstimate.inline_cold')}
      </span>
    );
  }
  const time = formatRenderDuration(estimate.seconds, i18n.language);
  const range = estimateRange(estimate, t, i18n.language) ?? '';
  const rough = estimate.basis === 'rough';
  return (
    <span
      data-slot="render-estimate"
      title={rough ? `${t('renderEstimate.rough_hint')} ${range}`.trim() : range || undefined}
    >
      {' · '}
      {rough
        ? t('renderEstimate.inline_roughly', { time })
        : t('renderEstimate.inline_about', { time })}
    </span>
  );
}
