import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import WaveSurfer from 'wavesurfer.js';
import Regions, { type Region } from 'wavesurfer.js/dist/plugins/regions.esm.js';
import { apiFetch, describeError } from '@/lib/api/client';
import { waveColors } from '@/components/waveform-player';
import { decodeToMonoLowRate } from '../../../../../../frontend/src/utils/audioTrim';
import {
  clampRange,
  defaultWindow,
  presetRange,
  rmsEnvelope,
  snapRange,
  TRIM_MIN_SECONDS,
  type Envelope,
  type TrimRange,
  type TrimSide,
} from './trim-math';

const REGION_COLOR = 'color-mix(in srgb, var(--primary) 20%, transparent)';
const MAX_ZOOM_PX_PER_SECOND = 2000;
const SETTLE_EPSILON = 1e-4;

export interface TrimWaveformOptions {
  /** A backend path (fetched through the API client) or the clip itself. */
  source: string | Blob;
  maxSeconds: number;
  /** Length of the window the trimmer opens with. */
  defaultLength: number;
  /** Snap moved edges into pauses. Read live; changing it never reloads the clip. */
  snap: boolean;
  busy: boolean;
  /** Mono decode rate; the exported WAV uses it too. */
  decodeSampleRate?: number;
  height?: number;
}

export interface TrimWaveform {
  container: RefObject<HTMLDivElement | null>;
  buffer: RefObject<AudioBuffer | null>;
  duration: number;
  /** The settled selection: what the preview and the export use. */
  range: TrimRange;
  /** The selection while a drag is in progress, for the readout only. */
  live: TrimRange | null;
  error: string;
  dismissError(): void;
  /** Set the selection; `side` says what moved so snapping only touches that, null = exact. */
  apply(next: TrimRange, side: TrimSide | null): void;
  /** Keep the start, set the length (slides back near the clip end). */
  preset(lengthSeconds: number): void;
  zoomIn(): void;
  zoomOut(): void;
  fitAll(): void;
  fitSelection(): void;
}

/**
 * Decodes the clip, draws it with wavesurfer and keeps exactly one region on
 * it: the selection. Dragging on empty waveform draws a new selection (the old
 * one goes), dragging the region moves it, its handles resize it, and every
 * change lands through `settle`, which clamps to the cap and snaps into pauses
 * when snapping is on.
 */
export function useTrimWaveform({
  source,
  maxSeconds,
  defaultLength,
  snap,
  busy,
  decodeSampleRate = 22050,
  height = 130,
}: TrimWaveformOptions): TrimWaveform {
  const container = useRef<HTMLDivElement>(null);
  const buffer = useRef<AudioBuffer | null>(null);
  const envelope = useRef<Envelope | null>(null);
  const waveform = useRef<WaveSurfer | null>(null);
  const region = useRef<Region | null>(null);
  const zoom = useRef(0);
  const snapRef = useRef(snap);
  const busyRef = useRef(busy);
  const rangeRef = useRef<TrimRange>({ start: 0, end: 0 });
  const [duration, setDuration] = useState(0);
  const [range, setRange] = useState<TrimRange>({ start: 0, end: 0 });
  const [live, setLive] = useState<TrimRange | null>(null);
  const [error, setError] = useState('');
  snapRef.current = snap;
  busyRef.current = busy;

  const settle = useCallback(
    (next: TrimRange, side: TrimSide | null) => {
      const clipDuration = buffer.current?.duration ?? 0;
      const fixed =
        side && snapRef.current && envelope.current
          ? snapRange(envelope.current, next, side, { maxSeconds })
          : clampRange(next, clipDuration, maxSeconds);
      const current = region.current;
      if (
        current &&
        (Math.abs(current.start - fixed.start) > SETTLE_EPSILON ||
          Math.abs(current.end - fixed.end) > SETTLE_EPSILON)
      )
        current.setOptions(fixed);
      rangeRef.current = fixed;
      setRange(fixed);
    },
    [maxSeconds],
  );

  useEffect(() => {
    const controller = new AbortController();
    let disposed = false;
    let disableDragSelection: (() => void) | null = null;
    setDuration(0);
    setError('');
    void (async () => {
      try {
        const blob =
          typeof source === 'string'
            ? await (await apiFetch(source, { signal: controller.signal })).blob()
            : source;
        const decoded = await decodeToMonoLowRate(blob, decodeSampleRate);
        if (disposed || !container.current) return;
        if (!Number.isFinite(decoded.duration) || decoded.duration < 0.02)
          throw new Error('Audio too short');
        buffer.current = decoded;
        const samples = decoded.getChannelData(0);
        envelope.current = rmsEnvelope(samples, decoded.sampleRate);
        const regions = Regions.create();
        const wave = WaveSurfer.create({
          container: container.current,
          peaks: [samples],
          duration: decoded.duration,
          height,
          ...waveColors(container.current),
          cursorWidth: 0,
          barWidth: 2,
          barGap: 1,
          barRadius: 2,
          normalize: true,
          interact: false,
          plugins: [regions],
        });
        waveform.current = wave;
        wave.on('error', (failure) => {
          if (!disposed) setError(describeError(failure));
        });
        const initial = defaultWindow(envelope.current, defaultLength, {
          maxSeconds,
          snap: snapRef.current,
        });
        wave.once('ready', () => {
          if (disposed) return;
          const params = {
            minLength: TRIM_MIN_SECONDS,
            maxLength: maxSeconds,
            drag: !busyRef.current,
            resize: !busyRef.current,
            color: REGION_COLOR,
          };
          region.current = regions.addRegion({ ...initial, ...params });
          disableDragSelection = regions.enableDragSelection(params);
          regions.on('region-created', (created) => {
            if (created === region.current) return;
            if (busyRef.current) {
              created.remove();
              return;
            }
            // A selection drawn on empty waveform replaces the current one.
            region.current?.remove();
            region.current = created;
            settle({ start: created.start, end: created.end }, 'edges');
          });
          regions.on('region-update', (updated) => {
            if (updated === region.current) setLive({ start: updated.start, end: updated.end });
          });
          regions.on('region-updated', (updated, side) => {
            if (updated !== region.current) return;
            setLive(null);
            settle({ start: updated.start, end: updated.end }, side ?? 'move');
          });
          setDuration(decoded.duration);
          settle(initial, null);
        });
      } catch (failure) {
        if (!disposed) setError(describeError(failure));
      }
    })();
    return () => {
      disposed = true;
      controller.abort();
      disableDragSelection?.();
      waveform.current?.destroy();
      waveform.current = null;
      region.current = null;
      buffer.current = null;
      envelope.current = null;
      zoom.current = 0;
    };
  }, [source, decodeSampleRate, maxSeconds, defaultLength, height, settle]);

  useEffect(() => {
    region.current?.setOptions({ drag: !busy, resize: !busy });
  }, [busy, duration]);

  const magnify = useCallback((factor: number) => {
    const clipDuration = buffer.current?.duration ?? 0;
    if (!clipDuration) return;
    const fit = (container.current?.clientWidth || 600) / clipDuration;
    zoom.current = Math.min(MAX_ZOOM_PX_PER_SECOND, Math.max(fit, (zoom.current || fit) * factor));
    waveform.current?.zoom(zoom.current);
  }, []);

  return {
    container,
    buffer,
    duration,
    range,
    live,
    error,
    dismissError: useCallback(() => setError(''), []),
    apply: settle,
    preset: useCallback(
      (lengthSeconds: number) => {
        const clipDuration = buffer.current?.duration ?? 0;
        if (!clipDuration) return;
        settle(presetRange(rangeRef.current.start, lengthSeconds, clipDuration), 'both');
      },
      [settle],
    ),
    zoomIn: useCallback(() => magnify(2), [magnify]),
    zoomOut: useCallback(() => magnify(0.5), [magnify]),
    fitAll: useCallback(() => {
      zoom.current = 0;
      waveform.current?.zoom(0);
    }, []),
    fitSelection: useCallback(() => {
      const { start, end } = rangeRef.current;
      zoom.current = (container.current?.clientWidth || 600) / Math.max(0.02, end - start);
      waveform.current?.zoom(zoom.current);
      waveform.current?.setScrollTime(start);
    }, []),
  };
}
