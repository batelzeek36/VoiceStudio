import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { MaximizeIcon, RepeatIcon, ZoomInIcon, ZoomOutIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Kbd } from '@/components/ui/kbd';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { PipelineFailure } from '@/components/pipeline-failure';
import { WaveformPlayer } from '@/components/waveform-player';
import { cn } from '@/lib/utils';
import {
  formatTrimClock,
  trimFit,
  TRIM_BEST_MAX_SECONDS,
  TRIM_BEST_MIN_SECONDS,
  TRIM_MIN_SECONDS,
  TRIM_SHORT_SECONDS,
  type TrimFit,
  type TrimRange,
} from './trim-math';
import { trimmedAudio, trimmedFileName } from './trimmed-audio';
import { useTrimWaveform } from './use-trim-waveform';

export const DEFAULT_TRIM_PRESETS = [10, 15, 20] as const;

export interface AudioTrimmerProps {
  /** A backend path or the clip itself. */
  source: string | Blob;
  /** Names the exported file (extension replaced, cut appended). */
  name: string;
  maxSeconds: number;
  /** One-click lengths; keys 1, 2, 3 pick them in order. */
  presets?: readonly number[];
  /** Length of the window the trimmer opens with. */
  defaultLength?: number;
  decodeSampleRate?: number;
  busy?: boolean;
  saveError?: string;
  confirmLabel?: string;
  /** Vidstack source key, so two trimmers never share a playback clock. */
  playerSource?: string;
  onSave(file: File, range: TrimRange): void;
  onCancel(): void;
  className?: string;
}

const FIT_CLASS: Record<TrimFit, string> = {
  best: 'text-success',
  ok: 'text-foreground',
  short: 'text-warning-foreground',
  long: 'text-destructive',
};

/**
 * Waveform + one selection with length presets, silence snapping, zoom and a
 * looping preview of exactly what will be exported. Headerless on purpose: the
 * gallery page and the clone dialog each frame it their own way.
 */
export function AudioTrimmer({
  source,
  name,
  maxSeconds,
  presets = DEFAULT_TRIM_PRESETS,
  defaultLength = 15,
  decodeSampleRate,
  busy = false,
  saveError,
  confirmLabel,
  playerSource = 'audio-trimmer',
  onSave,
  onCancel,
  className,
}: AudioTrimmerProps) {
  const { t } = useTranslation();
  const section = useRef<HTMLElement>(null);
  const previewContainer = useRef<HTMLDivElement>(null);
  const [snap, setSnap] = useState(true);
  const [loop, setLoop] = useState(true);
  const [preview, setPreview] = useState<{ blob: Blob; url: string } | null>(null);
  const snapId = useId();
  const wave = useTrimWaveform({
    source,
    maxSeconds,
    defaultLength,
    snap,
    busy,
    decodeSampleRate,
  });
  const { duration, range, live, error, buffer } = wave;
  const shown = live ?? range;
  const length = shown.end - shown.start;
  const fit = trimFit(length, maxSeconds);
  const ready = duration > 0;

  useEffect(() => {
    if (!buffer.current || !duration) return;
    try {
      const blob = trimmedAudio(buffer.current, range.start, range.end, maxSeconds);
      const url = URL.createObjectURL(blob);
      setPreview({ blob, url });
      return () => URL.revokeObjectURL(url);
    } catch {
      setPreview(null);
    }
  }, [buffer, range, duration, maxSeconds]);

  // Keys work as soon as the clip is on screen, without a click first.
  useEffect(() => {
    if (ready) section.current?.focus({ preventScroll: true });
  }, [ready]);

  const canSave = Boolean(preview) && !busy && !error && fit !== 'long';
  const save = () => {
    if (!preview || !canSave) return;
    onSave(
      new File([preview.blob], trimmedFileName(name, range.start, range.end), {
        type: 'audio/wav',
      }),
      range,
    );
  };

  const change = (key: 'start' | 'end', value: number) => {
    if (!Number.isFinite(value)) return;
    const next =
      key === 'start'
        ? { start: Math.max(0, Math.min(value, range.end - TRIM_MIN_SECONDS)), end: range.end }
        : {
            start: range.start,
            end: Math.min(duration, Math.max(value, range.start + TRIM_MIN_SECONDS)),
          };
    if (next.end - next.start > maxSeconds) {
      if (key === 'start') next.end = next.start + maxSeconds;
      else next.start = next.end - maxSeconds;
    }
    // Typed numbers are exact: no snapping away from what was entered.
    wave.apply(next, null);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    const target = event.target as HTMLElement;
    if (event.key === 'Escape' && !busy) {
      event.preventDefault();
      event.stopPropagation();
      onCancel();
      return;
    }
    const typing = Boolean(target.closest('input,textarea'));
    if (!typing && ready && !busy && /^[1-9]$/.test(event.key)) {
      const preset = presets[Number(event.key) - 1];
      if (preset !== undefined) {
        event.preventDefault();
        wave.preset(preset);
      }
      return;
    }
    if (typing || target.closest('button')) return;
    if (event.key === ' ' && preview) {
      event.preventDefault();
      event.stopPropagation();
      previewContainer.current?.querySelector('button')?.click();
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      event.stopPropagation();
      save();
    }
  };

  const fitNote =
    fit === 'best'
      ? t('trimmer.fit_best')
      : fit === 'short'
        ? t('trimmer.fit_short', { seconds: TRIM_SHORT_SECONDS })
        : fit === 'long'
          ? t('trimmer.fit_long', { max: maxSeconds })
          : t('trimmer.fit_ok', { min: TRIM_BEST_MIN_SECONDS, max: TRIM_BEST_MAX_SECONDS });

  return (
    <section
      ref={section}
      tabIndex={-1}
      data-slot="audio-trimmer"
      onKeyDown={onKeyDown}
      className={cn('flex flex-col gap-4 outline-none', className)}
    >
      {saveError && (
        <p role="alert" className="text-sm text-destructive">
          {saveError}
        </p>
      )}
      {error && (
        <PipelineFailure
          fallback={`${t('trimmer.audio_load_failed')} ${error}`}
          onDismiss={wave.dismissError}
        />
      )}
      {!ready && !error && (
        <p role="status" className="text-sm text-muted-foreground">
          {t('trimmer.decoding')}
        </p>
      )}
      <div
        ref={wave.container}
        data-slot="audio-trimmer-wave"
        className={cn(
          'overflow-hidden rounded-xl border border-border/50 bg-muted/20 p-3 text-muted-foreground',
          !ready && 'min-h-38',
        )}
      />
      {ready && (
        <>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <div
              role="group"
              aria-label={t('trimmer.presets_label')}
              className="flex items-center gap-1.5"
            >
              <span className="text-[length:var(--text-label)] text-muted-foreground">
                {t('trimmer.presets_label')}
              </span>
              {presets.map((seconds, index) => (
                <Button
                  key={seconds}
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={busy}
                  className="gap-2 tabular-nums"
                  onClick={() => wave.preset(seconds)}
                >
                  {t('trimmer.preset_seconds', { seconds })}
                  {index < 9 && <Kbd aria-hidden="true">{index + 1}</Kbd>}
                </Button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <Switch
                id={snapId}
                size="sm"
                checked={snap}
                disabled={busy}
                onCheckedChange={setSnap}
              />
              <Label
                htmlFor={snapId}
                className="text-[length:var(--text-label)] font-normal text-muted-foreground"
              >
                {t('trimmer.snap_label')}
              </Label>
            </div>
            <div className="ml-auto flex items-center gap-1">
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                aria-label={t('trimmer.zoom_in')}
                onClick={wave.zoomIn}
              >
                <ZoomInIcon />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                aria-label={t('trimmer.zoom_out')}
                onClick={wave.zoomOut}
              >
                <ZoomOutIcon />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                aria-label={t('trimmer.fit_all')}
                onClick={wave.fitAll}
              >
                <MaximizeIcon />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                aria-label={t('trimmer.fit_selection')}
                onClick={wave.fitSelection}
              >
                {t('trimmer.fit_sel_btn')}
              </Button>
              <span className="ml-1 text-xs tabular-nums text-muted-foreground">
                {duration.toFixed(2)} {t('trimmer.unit_seconds')}
              </span>
            </div>
          </div>
          <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
            <div className="flex min-w-0 flex-col gap-0.5">
              <p
                role="status"
                aria-live="polite"
                className={cn('text-lg font-medium tabular-nums', FIT_CLASS[fit])}
              >
                {t('trimmer.readout', {
                  start: formatTrimClock(shown.start),
                  end: formatTrimClock(shown.end),
                  length: length.toFixed(1),
                })}
              </p>
              <p className="text-[length:var(--text-caption)] text-muted-foreground">{fitNote}</p>
            </div>
            <div className="flex items-end gap-3">
              {(['start', 'end'] as const).map((key) => (
                <label key={key} className="flex flex-col gap-1.5 text-xs">
                  <span className="text-muted-foreground">{t('trimmer.' + key + '_label')}</span>
                  <Input
                    type="number"
                    min={0}
                    max={duration}
                    step="0.01"
                    value={Number(range[key].toFixed(2))}
                    disabled={busy}
                    onChange={(event) => change(key, event.target.valueAsNumber)}
                    className="w-28 tabular-nums"
                  />
                </label>
              ))}
            </div>
          </div>
          {preview && (
            <div ref={previewContainer} className="flex items-center gap-2">
              <WaveformPlayer
                key={preview.url}
                src={preview.url}
                source={playerSource}
                loop={loop}
                className="flex-1"
              />
              <Button
                type="button"
                variant={loop ? 'secondary' : 'ghost'}
                size="icon-sm"
                aria-label={t('trimmer.loop_preview')}
                aria-pressed={loop}
                onClick={() => setLoop(!loop)}
              >
                <RepeatIcon />
              </Button>
            </div>
          )}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[length:var(--text-caption)] text-muted-foreground">
              <span>{t('trimmer.hint_drag')}</span>
              <span className="inline-flex items-center gap-1">
                <Kbd>Space</Kbd> {t('trimmer.hint_play')}
              </span>
              <span className="inline-flex items-center gap-1">
                <Kbd>1</Kbd>
                <Kbd>2</Kbd>
                <Kbd>3</Kbd> {t('trimmer.hint_length')}
              </span>
              <span className="inline-flex items-center gap-1">
                <Kbd>Enter</Kbd> {t('trimmer.hint_use')}
              </span>
              <span className="inline-flex items-center gap-1">
                <Kbd>Esc</Kbd> {t('trimmer.hint_cancel')}
              </span>
            </p>
            <div className="ml-auto flex gap-2">
              <Button type="button" variant="ghost" disabled={busy} onClick={onCancel}>
                {t('common.cancel')}
              </Button>
              <Button type="button" disabled={!canSave} onClick={save}>
                {busy ? t('common.loading') : (confirmLabel ?? t('trimmer.use_trimmed'))}
              </Button>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
