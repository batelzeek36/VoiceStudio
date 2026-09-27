import type { TFunction } from 'i18next';
import { TimerIcon } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGenerateClone } from '@/hooks/use-generate';
import { useRenderEstimate } from '@/hooks/use-render-estimate';
import { cloneEstimateRequest, type RenderEstimate } from '@/lib/api/render-estimate';
import { formatRenderDuration, takeRemaining } from '@/lib/render-estimate-time';
import { useCloneSettings } from '@/lib/store/clone-settings';
import { useReference } from '@/lib/store/reference';

const TICK_MS = 1000;

/** The idle line: the estimate, rough estimate or calibration note. */
export function estimateLine(
  estimate: RenderEstimate | null,
  t: TFunction,
  locale: string,
): { text: string; title?: string } | null {
  if (!estimate) return null;
  if (estimate.seconds === null) {
    return estimate.reason === 'cold_start' ? { text: t('renderEstimate.cold') } : null;
  }
  const time = formatRenderDuration(estimate.seconds, locale);
  const range =
    estimate.low !== null && estimate.high !== null
      ? t('renderEstimate.range', {
          low: formatRenderDuration(estimate.low, locale),
          high: formatRenderDuration(estimate.high, locale),
        })
      : undefined;
  if (estimate.basis === 'rough') {
    return {
      text: t('renderEstimate.roughly', { time }),
      title: [t('renderEstimate.rough_hint'), range].filter(Boolean).join(' '),
    };
  }
  return { text: t('renderEstimate.about', { time }), title: range };
}

/**
 * Beside Synthesize: how long this take will render on this machine, then a
 * countdown while it renders. The estimate is frozen when the render starts so
 * editing the script mid-render cannot move the countdown.
 */
export function RenderEstimateHint() {
  const { t, i18n } = useTranslation();
  const settings = useCloneSettings();
  const reference = useReference();
  const { isGenerating, stage } = useGenerateClone();
  const request = useMemo(
    () => cloneEstimateRequest(settings, reference.durationSeconds),
    [settings, reference.durationSeconds],
  );
  const estimate = useRenderEstimate(request);
  const [frozen, setFrozen] = useState<{ estimate: RenderEstimate | null } | null>(null);
  if (isGenerating && frozen === null) setFrozen({ estimate });
  if (!isGenerating && frozen !== null) setFrozen(null);

  // The countdown starts once the engine is synthesizing, not while the model
  // is still loading: the estimate prices synthesis only.
  const rendering = isGenerating && (stage === 'generating' || stage === 'receiving');
  const [clock, setClock] = useState<{ started: number; now: number } | null>(null);
  useEffect(() => {
    if (!rendering) return;
    const started = Date.now();
    const tick = () => setClock({ started, now: Date.now() });
    tick();
    const timer = window.setInterval(tick, TICK_MS);
    return () => {
      window.clearInterval(timer);
      setClock(null);
    };
  }, [rendering]);

  let line: { text: string; title?: string } | null = null;
  if (isGenerating) {
    const remaining =
      clock && frozen?.estimate
        ? takeRemaining(frozen.estimate.seconds, (clock.now - clock.started) / 1000)
        : null;
    if (remaining !== null) {
      line = {
        text:
          remaining > 0
            ? t('renderEstimate.left', { time: formatRenderDuration(remaining, i18n.language) })
            : t('renderEstimate.finishing'),
      };
    }
  } else if (request) {
    line = estimateLine(estimate, t, i18n.language);
  }
  if (!line) return null;
  return (
    <p
      data-slot="render-estimate"
      title={line.title}
      className="flex max-w-56 min-w-0 items-center justify-end gap-1.5 text-right text-xs text-muted-foreground tabular-nums"
    >
      <TimerIcon className="size-3.5 shrink-0" aria-hidden="true" />
      <span className="min-w-0">{line.text}</span>
    </p>
  );
}
