import {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from 'react';
import { useTranslation } from 'react-i18next';
import {
  ChevronDownIcon,
  FocusIcon,
  MaximizeIcon,
  PauseIcon,
  PlayIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Input } from '@/components/ui/input';
import { Kbd } from '@/components/ui/kbd';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { PipelineFailure } from '@/components/pipeline-failure';
import { describeError } from '@/lib/api/client';
import { cn } from '@/lib/utils';
import {
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
  /** Reopen on a known cut instead of the default window. */
  initialRange?: TrimRange | null;
  decodeSampleRate?: number;
  /** Offer zoom under Fine-tune (long recordings); keys + - Home End then work. */
  zoom?: boolean;
  busy?: boolean;
  saveError?: string;
  confirmLabel?: string;
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
 * One waveform, one selection, one job: pick the stretch to keep. Visible:
 * the waveform (drag to select, playhead runs on it), Play, the length
 * presets, the length readout, Cancel and the confirm. Everything finer
 * (exact seconds, snapping, loop, zoom, the keys) sits under Fine-tune.
 * Headerless on purpose: the gallery page and the clone dialog frame it.
 */
export function AudioTrimmer({
  source,
  name,
  maxSeconds,
  presets = DEFAULT_TRIM_PRESETS,
  defaultLength = 15,
  initialRange = null,
  decodeSampleRate,
  zoom = false,
  busy = false,
  saveError,
  confirmLabel,
  onSave,
  onCancel,
  className,
}: AudioTrimmerProps) {
  const { t } = useTranslation();
  const [snap, setSnap] = useState(true);
  const [loop, setLoop] = useState(true);
  const [fineTune, setFineTune] = useState(false);
  const [exportError, setExportError] = useState('');
  const snapId = useId();
  const loopId = useId();
  const wave = useTrimWaveform({
    source,
    maxSeconds,
    defaultLength,
    initialRange,
    snap,
    loop,
    busy,
    decodeSampleRate,
  });
  const { duration, range, live, error, buffer, playing } = wave;
  const shown = live ?? range;
  const length = shown.end - shown.start;
  const fit = trimFit(length, maxSeconds);
  const ready = duration > 0;
  const canSave = ready && !busy && !error && fit !== 'long';

  const save = () => {
    const decoded = buffer.current;
    if (!decoded || !canSave) return;
    let blob: Blob;
    try {
      blob = trimmedAudio(decoded, range.start, range.end, maxSeconds);
    } catch (failure) {
      setExportError(describeError(failure));
      return;
    }
    onSave(
      new File([blob], trimmedFileName(name, range.start, range.end), { type: 'audio/wav' }),
      range,
    );
  };

  const commitField = (key: 'start' | 'end', value: number) => {
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
    // Typed seconds are exact: no snapping away from what was entered.
    wave.apply(next, null);
  };

  // Keys are handled ahead of everything else while the trimmer is up, so a
  // focused button or the media element can never swallow them. Typing in a
  // field keeps its keys; a focused switch keeps Space.
  const handlers = useRef({
    cancel: onCancel,
    save,
    toggle: wave.togglePlay,
    preset: wave.preset,
    zoomIn: wave.zoomIn,
    zoomOut: wave.zoomOut,
    fitAll: wave.fitAll,
    fitSelection: wave.fitSelection,
  });
  handlers.current = {
    cancel: onCancel,
    save,
    toggle: wave.togglePlay,
    preset: wave.preset,
    zoomIn: wave.zoomIn,
    zoomOut: wave.zoomOut,
    fitAll: wave.fitAll,
    fitSelection: wave.fitSelection,
  };
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target instanceof HTMLElement ? event.target : null;
      const handled = () => {
        event.preventDefault();
        event.stopPropagation();
      };
      if (event.key === 'Escape') {
        if (busy) return;
        handled();
        handlers.current.cancel();
        return;
      }
      if (target?.closest('input,textarea,select,[contenteditable="true"]')) return;
      if (!ready || busy) return;
      if (/^[1-9]$/.test(event.key)) {
        const preset = presets[Number(event.key) - 1];
        if (preset === undefined) return;
        handled();
        handlers.current.preset(preset);
        return;
      }
      if (event.key === ' ') {
        if (target?.closest('[role="switch"],[role="checkbox"]')) return;
        handled();
        handlers.current.toggle();
        return;
      }
      if (event.key === 'Enter') {
        // Only Cancel and the confirm keep their own Enter; a preset, Play or
        // Fine-tune left focused by a click must not re-fire instead.
        if (target?.closest('[data-slot="audio-trimmer-footer"],a')) return;
        handled();
        handlers.current.save();
        return;
      }
      if (!zoom) return;
      if (event.key === '+' || event.key === '=') {
        handled();
        handlers.current.zoomIn();
      } else if (event.key === '-') {
        handled();
        handlers.current.zoomOut();
      } else if (event.key === 'Home') {
        handled();
        handlers.current.fitAll();
      } else if (event.key === 'End') {
        handled();
        handlers.current.fitSelection();
      }
    };
    window.addEventListener('keydown', onKeyDown, true);
    return () => window.removeEventListener('keydown', onKeyDown, true);
  }, [busy, presets, ready, zoom]);

  const fitNote =
    fit === 'best'
      ? t('trimmer.fit_best')
      : fit === 'short'
        ? t('trimmer.fit_short', { seconds: TRIM_SHORT_SECONDS })
        : fit === 'long'
          ? t('trimmer.fit_long', { max: maxSeconds })
          : t('trimmer.fit_ok', { min: TRIM_BEST_MIN_SECONDS, max: TRIM_BEST_MAX_SECONDS });
  const failure = saveError || exportError;

  return (
    <section
      data-slot="audio-trimmer"
      className={cn('flex min-w-0 flex-col gap-4 [container-type:inline-size]', className)}
    >
      {failure && (
        <p role="alert" className="text-sm text-destructive">
          {failure}
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
          'w-full min-w-0 max-w-full overflow-hidden rounded-xl border border-border/50 bg-muted/20 p-3 text-muted-foreground',
          !ready && 'min-h-40',
        )}
      />
      {ready && (
        <>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
            <Button
              type="button"
              variant="secondary"
              size="sm"
              disabled={busy}
              aria-pressed={playing}
              onClick={wave.togglePlay}
            >
              {playing ? (
                <PauseIcon data-icon="inline-start" />
              ) : (
                <PlayIcon data-icon="inline-start" />
              )}
              {t(playing ? 'player.pause' : 'player.play')}
            </Button>
            <div
              role="group"
              aria-label={t('trimmer.presets_label')}
              className="flex items-center gap-1.5"
            >
              <span className="text-[length:var(--text-label)] text-muted-foreground">
                {t('trimmer.presets_label')}
              </span>
              {presets.map((seconds) => (
                <Button
                  key={seconds}
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={busy}
                  className="tabular-nums"
                  onClick={() => wave.preset(seconds)}
                >
                  {t('trimmer.preset_seconds', { seconds })}
                </Button>
              ))}
            </div>
            <div className="ml-auto flex min-w-0 flex-col items-end">
              <p className={cn('text-lg font-medium tabular-nums', FIT_CLASS[fit])}>
                {t('trimmer.length_seconds', { length: length.toFixed(1) })}
              </p>
              <p className="text-[length:var(--text-caption)] text-muted-foreground">{fitNote}</p>
            </div>
          </div>
          <Collapsible
            open={fineTune}
            onOpenChange={setFineTune}
            className="flex flex-col items-start gap-2"
          >
            <CollapsibleTrigger className="inline-flex items-center gap-1 rounded-sm text-[length:var(--text-caption)] text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50">
              {t('trimmer.fine_tune')}
              <ChevronDownIcon
                className={cn('size-3 transition-transform', fineTune && 'rotate-180')}
                aria-hidden="true"
              />
            </CollapsibleTrigger>
            <CollapsibleContent className="w-full">
              <div className="flex flex-col gap-3 rounded-lg bg-muted/30 p-3">
                <div className="flex flex-wrap items-end gap-x-3 gap-y-2">
                  {(['start', 'end'] as const).map((key) => (
                    <SecondsField
                      key={key}
                      label={t('trimmer.' + key + '_label')}
                      value={range[key]}
                      max={duration}
                      disabled={busy}
                      onCommit={(value) => commitField(key, value)}
                    />
                  ))}
                  <span className="pb-2 text-[length:var(--text-caption)] tabular-nums text-muted-foreground">
                    {t('trimmer.of_total', { seconds: duration.toFixed(2) })}
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
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
                  <div className="flex items-center gap-2">
                    <Switch id={loopId} size="sm" checked={loop} onCheckedChange={setLoop} />
                    <Label
                      htmlFor={loopId}
                      className="text-[length:var(--text-label)] font-normal text-muted-foreground"
                    >
                      {t('trimmer.loop_preview')}
                    </Label>
                  </div>
                  {zoom && (
                    <div className="flex items-center gap-1">
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
                        size="icon-sm"
                        aria-label={t('trimmer.fit_selection')}
                        onClick={wave.fitSelection}
                      >
                        <FocusIcon />
                      </Button>
                    </div>
                  )}
                </div>
                <ul className="flex flex-wrap gap-x-5 gap-y-1.5 text-[length:var(--text-caption)] text-muted-foreground">
                  <li>{t('trimmer.keys_drag')}</li>
                  <li className="inline-flex items-center gap-1.5">
                    <Kbd>Space</Kbd> {t('trimmer.key_play')}
                  </li>
                  <li className="inline-flex items-center gap-1.5">
                    <Kbd>1</Kbd>
                    <Kbd>2</Kbd>
                    <Kbd>3</Kbd> {t('trimmer.key_length')}
                  </li>
                  <li className="inline-flex items-center gap-1.5">
                    <Kbd>Enter</Kbd> {t('trimmer.key_confirm')}
                  </li>
                  <li className="inline-flex items-center gap-1.5">
                    <Kbd>Esc</Kbd> {t('trimmer.key_cancel')}
                  </li>
                  {zoom && (
                    <>
                      <li className="inline-flex items-center gap-1.5">
                        <Kbd>+</Kbd>
                        <Kbd>-</Kbd> {t('trimmer.key_zoom')}
                      </li>
                      <li className="inline-flex items-center gap-1.5">
                        <Kbd>Home</Kbd>
                        <Kbd>End</Kbd> {t('trimmer.key_fit')}
                      </li>
                    </>
                  )}
                </ul>
              </div>
            </CollapsibleContent>
          </Collapsible>
          <div data-slot="audio-trimmer-footer" className="flex justify-end gap-2">
            <Button type="button" variant="ghost" disabled={busy} onClick={onCancel}>
              {t('common.cancel')}
            </Button>
            <Button type="button" disabled={!canSave} onClick={save}>
              {busy ? t('common.loading') : (confirmLabel ?? t('trimmer.use_trimmed'))}
            </Button>
          </div>
        </>
      )}
    </section>
  );
}

/** Seconds field that commits on blur or Enter, never per keystroke. */
function SecondsField({
  label,
  value,
  max,
  disabled,
  onCommit,
}: {
  label: string;
  value: number;
  max: number;
  disabled: boolean;
  onCommit(value: number): void;
}) {
  const [text, setText] = useState(value.toFixed(2));
  useEffect(() => setText(value.toFixed(2)), [value]);
  const commit = () => {
    const parsed = Number(text);
    if (Number.isFinite(parsed) && text.trim() !== '') onCommit(parsed);
    else setText(value.toFixed(2));
  };
  const onKeyDown = (event: ReactKeyboardEvent<HTMLInputElement>) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    event.stopPropagation();
    commit();
    event.currentTarget.blur();
  };
  return (
    <label className="flex flex-col gap-1.5 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <Input
        type="number"
        min={0}
        max={max}
        step="0.01"
        value={text}
        disabled={disabled}
        onChange={(event) => setText(event.target.value)}
        onBlur={commit}
        onKeyDown={onKeyDown}
        className="w-28 tabular-nums"
      />
    </label>
  );
}
