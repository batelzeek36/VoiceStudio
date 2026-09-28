import type { TFunction } from 'i18next';
import { ServerOffIcon, TimerIcon } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGenerateClone } from '@/hooks/use-generate';
import { useRenderEstimate } from '@/hooks/use-render-estimate';
import { cloneEstimateRequest, type RenderEstimate } from '@/lib/api/render-estimate';
import { estimateRange, remoteWorker, unavailableLine } from '@/lib/render-estimate-target';
import { formatRenderDuration, takeRemaining } from '@/lib/render-estimate-time';
import { useCloneSettings } from '@/lib/store/clone-settings';
import { useReference } from '@/lib/store/reference';

const TICK_MS = 1000;

/** The idle line: the estimate, rough estimate or calibration note. */
export function estimateLine(
  estimate: RenderEstimate | null,
  t: TFunction,
  locale: string,
): { text: string; title?: string; unavailable?: boolean } | null {
  if (!estimate) return null;
  const unavailable = unavailableLine(estimate, t);
  if (unavailable) return { text: unavailable, unavailable: true };
  if (estimate.seconds === null) {
    if (estimate.reason !== 'cold_start') return null;
    const worker = remoteWorker(estimate);
    return { text: worker ? t('renderEstimate.cold_on', { worker }) : t('renderEstimate.cold') };
  }
  const time = formatRenderDuration(estimate.seconds, locale);
  const range = estimateRange(estimate, t, locale);
  if (estimate.basis === 'rough') {
    return {
      text: t('renderEstimate.roughly', { time }),
      title: [t('renderEstimate.rough_hint'), range].filter(Boolean).join(' '),
    };
  }
  return { text: t('renderEstimate.about', { time }), title: range };
}

/**
 * Beside Synthesize: how long this take will render where it runs (this
 * machine, or the GPU worker the picker chose), then a countdown while it
 * renders. The estimate is frozen when the render starts so
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

  let line: { text: string; title?: string; unavailable?: boolean } | null = null;
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
      {line.unavailable ? (
        <ServerOffIcon className="size-3.5 shrink-0" aria-hidden="true" />
      ) : (
        <TimerIcon className="size-3.5 shrink-0" aria-hidden="true" />
      )}
      <span className="min-w-0">{line.text}</span>
    </p>
  );
}
