import { LoaderCircleIcon, MicIcon, SparklesIcon, SquareIcon, UploadCloudIcon } from 'lucide-react';
import { useId, useRef, useState, type DragEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { toast } from 'sonner';
import { RecordingInputs } from '@/components/recording-inputs';
import { useRecording } from '@/hooks/use-recording';
import { useEngines } from '@/hooks/use-engines';
import { REF_HARD_MAX_SECONDS, REF_TEXT_MAX_SECONDS } from '@/lib/api/generate';
import { probeAudioDuration } from '@/lib/audio/probe';
import { referenceUsageNote } from '@/lib/reference-usage';
import { setReferenceFile } from '@/lib/store/reference';
import { cn } from '@/lib/utils';
import { decodeToMonoLowRate } from '../../../../../../frontend/src/utils/audioTrim';

const ACCEPT = 'audio/*,.mp3,.wav,.m4a,.flac,.ogg,.aac,.webm';
const AUDIO_EXT = /\.(mp3|wav|m4a|flac|ogg|aac|webm)$/i;
const LEVEL_THRESHOLD = 0.025;

function isAudioFile(file: File): boolean {
  return file.type.startsWith('audio/') || AUDIO_EXT.test(file.name);
}

type IngestFn = (file: File | null) => Promise<void>;

/** Receives an accepted clip instead of the composer's shared reference. */
export type AcceptReference = (file: File, durationSeconds: number | null) => void;
/** Receives a clip that must be trimmed before it can be a reference. */
export type TrimRequest = (file: File, durationSeconds: number) => void;

/** Longer recordings are not decoded in the renderer at all: pick a shorter file. */
export const REF_TRIM_INPUT_MAX_SECONDS = 30 * 60;

/** Clips longer than the transcript limit go through the trimmer first. */
export function needsReferenceTrim(durationSeconds: number | null): boolean {
  return durationSeconds !== null && durationSeconds > REF_TEXT_MAX_SECONDS;
}

async function clipDuration(file: File): Promise<number | null> {
  const probed = await probeAudioDuration(file);
  if (probed !== null) return probed;
  // MediaRecorder WebM carries no duration header (Chromium reports Infinity),
  // so measure a low-rate decode instead; a file that cannot be decoded at all
  // stays unknown and the backend reports it.
  try {
    const decoded = await decodeToMonoLowRate(file, 8000);
    return Number.isFinite(decoded.duration) && decoded.duration > 0 ? decoded.duration : null;
  } catch {
    return null;
  }
}

/**
 * Validate + load a reference clip, surfacing the length checks as toasts.
 * Clips over REF_TEXT_MAX_SECONDS are handed to `onTrim` and never accepted
 * as they are. Without `onAccept` an accepted clip becomes the composer's
 * reference; with it (the saved-profile editor) the clip is handed back.
 */
function useIngest(onAccept: AcceptReference | undefined, onTrim: TrimRequest): IngestFn {
  const { t } = useTranslation();
  // Monotonic pick token: a slow probe for an earlier clip must never replace
  // (or toast over) a later one.
  const latestPick = useRef(0);
  return async (file) => {
    if (!file) return;
    const pick = ++latestPick.current;
    if (!isAudioFile(file)) {
      toast.error(t('clone.unsupported_audio'));
      return;
    }
    const durationSeconds = await clipDuration(file);
    if (pick !== latestPick.current) return;
    if (durationSeconds !== null && durationSeconds > REF_TRIM_INPUT_MAX_SECONDS) {
      toast.error(
        t('referenceTrim.input_too_long', {
          minutes: Math.round(durationSeconds / 60),
          max: REF_TRIM_INPUT_MAX_SECONDS / 60,
        }),
      );
      return;
    }
    if (durationSeconds !== null && needsReferenceTrim(durationSeconds)) {
      onTrim(file, durationSeconds);
      return;
    }
    if (onAccept) {
      onAccept(file, durationSeconds);
      return;
    }
    const result = await setReferenceFile(file, durationSeconds);
    if (pick !== latestPick.current) return;
    // Only a clip whose length could not be measured reaches the store's own
    // ceiling; an accepted long clip gets ReferenceUsageNote beside it (#2281).
    if (!result.ok) {
      const duration = Math.round(result.durationSeconds ?? 0);
      toast.error(t('tts_errors.too_long', { duration, max: REF_HARD_MAX_SECONDS }));
    }
  };
}

/** How much of a clip longer than the 5–15 s recommendation the active engine uses. */
export function ReferenceUsageNote({ durationSeconds }: { durationSeconds: number | null }) {
  const { t } = useTranslation();
  const { activeTts } = useEngines();
  const note = referenceUsageNote(activeTts, durationSeconds);
  if (!note) return null;
  return (
    <p className="text-[length:var(--text-caption)] text-muted-foreground" role="status">
      {note.kind === 'best_window'
        ? t('clone.ref_usage_best_window', { seconds: note.seconds })
        : note.kind === 'head'
          ? t('clone.ref_usage_head', { seconds: note.seconds })
          : t('clone.ref_usage_long', { seconds: note.seconds })}
    </p>
  );
}

export interface ReferenceZoneProps {
  onAccept?: AcceptReference;
  /** Every entry point must say where an over-long clip goes to be trimmed. */
  onTrim: TrimRequest;
}

export function UploadZone({ onAccept, onTrim }: ReferenceZoneProps) {
  const { t } = useTranslation();
  const ingestFile = useIngest(onAccept, onTrim);
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const id = useId();

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    void ingestFile(event.dataTransfer.files[0] ?? null);
  };

  return (
    <div>
      <input
        ref={inputRef}
        id={id}
        type="file"
        accept={ACCEPT}
        className="sr-only"
        onChange={(event) => {
          void ingestFile(event.target.files?.[0] ?? null);
          event.target.value = '';
        }}
      />
      <label
        htmlFor={id}
        className={cn(
          'flex min-h-36 cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed bg-muted/30 px-4 py-6 text-center transition-colors hover:border-primary/60 hover:bg-muted/50 focus-within:ring-3 focus-within:ring-ring/50',
          dragging && 'border-primary bg-primary/10',
        )}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <UploadCloudIcon className="size-6 text-muted-foreground" aria-hidden="true" />
        <span className="text-[length:var(--text-label)] font-medium text-muted-foreground">
          {t('clone.drop_audio')}
        </span>
      </label>
    </div>
  );
}

export function RecordZone({ onAccept, onTrim }: ReferenceZoneProps) {
  const { t } = useTranslation();
  const ingestFile = useIngest(onAccept, onTrim);
  const rec = useRecording((file) => void ingestFile(file));
  const hasSignal = rec.level >= LEVEL_THRESHOLD;
  let micButton;
  if (rec.isStarting || rec.isCleaning) {
    micButton = (
      <div
        className="flex size-24 flex-col items-center justify-center gap-1.5 rounded-full bg-muted text-[length:var(--text-label)] font-medium text-muted-foreground"
        role="status"
        aria-live="polite"
      >
        {rec.isStarting ? (
          <LoaderCircleIcon className="size-5 animate-spin motion-reduce:animate-none" />
        ) : (
          <SparklesIcon className="size-5 animate-pulse motion-reduce:animate-none" />
        )}
        {rec.isStarting ? t('clone.starting_recording') : t('clone.cleaning')}
      </div>
    );
  } else if (rec.isRecording) {
    micButton = (
      <button
        type="button"
        onClick={() => rec.stop()}
        aria-label={t('clone.stop_recording')}
        className="relative flex size-24 flex-col items-center justify-center gap-1.5 rounded-full border-2 border-destructive bg-destructive/10 text-[length:var(--text-label)] font-semibold text-destructive outline-none focus-visible:ring-3 focus-visible:ring-destructive/40"
      >
        <span
          className="absolute inset-0 rounded-full border-2 border-destructive/60 animate-ping motion-reduce:animate-none"
          aria-hidden="true"
        />
        <SquareIcon className="size-5 fill-current" />
        <span className="tabular-nums">
          {t('clone.duration_seconds', { seconds: rec.seconds })}
        </span>
      </button>
    );
  } else {
    micButton = (
      <button
        type="button"
        onClick={() => void rec.start()}
        className="flex size-24 flex-col items-center justify-center gap-1.5 rounded-full bg-muted text-[length:var(--text-label)] font-medium text-muted-foreground transition-colors outline-none hover:bg-destructive/10 hover:text-destructive focus-visible:ring-3 focus-visible:ring-ring/50"
      >
        <MicIcon className="size-5" />
        {t('clone.record')}
      </button>
    );
  }

  return (
    <div className="flex flex-col gap-4 rounded-lg bg-muted/30 p-4">
      <div className="flex justify-center py-2">{micButton}</div>
      <RecordingInputs rec={rec} />
      {rec.isRecording ? (
        <div
          className="flex items-center gap-2 text-[length:var(--text-label)]"
          role="status"
          aria-live="polite"
        >
          <span
            className={cn(
              'size-2 shrink-0 rounded-full',
              hasSignal ? 'bg-success' : 'bg-muted-foreground',
            )}
            aria-hidden="true"
          />
          <div
            className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-muted"
            role="meter"
            aria-label={t('recording.input_level')}
            aria-valuemin={0}
            aria-valuemax={1}
            aria-valuenow={Number(rec.level.toFixed(3))}
          >
            <div
              className="h-full rounded-full bg-success transition-[width] duration-75 motion-reduce:transition-none"
              style={{ width: `${Math.min(100, Math.round(rec.level * 100))}%` }}
            />
          </div>
          <span className={hasSignal ? 'text-success' : 'text-muted-foreground'}>
            {hasSignal ? t('recording.input_detected') : t('recording.no_input_detected')}
          </span>
        </div>
      ) : null}
    </div>
  );
}
