import { useTranslation } from 'react-i18next';
import type { RenderEstimate } from '@/lib/api/render-estimate';
import { formatRenderDuration } from '@/lib/render-estimate-time';

/**
 * After the runtime estimate in the script stats line: how long rendering it
 * will take on this machine ("render about 13 min"), or when that number will
 * appear. Nothing for a remote render or while the first estimate loads.
 */
export function RenderEstimateInline({ estimate }: { estimate: RenderEstimate | null }) {
  const { t, i18n } = useTranslation();
  if (!estimate) return null;
  if (estimate.seconds === null) {
    if (estimate.reason !== 'cold_start') return null;
    return <span data-slot="render-estimate"> · {t('renderEstimate.inline_cold')}</span>;
  }
  const time = formatRenderDuration(estimate.seconds, i18n.language);
  const range =
    estimate.low !== null && estimate.high !== null
      ? t('renderEstimate.range', {
          low: formatRenderDuration(estimate.low, i18n.language),
          high: formatRenderDuration(estimate.high, i18n.language),
        })
      : '';
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
