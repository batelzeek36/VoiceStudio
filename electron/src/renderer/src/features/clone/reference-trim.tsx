import { useTranslation } from 'react-i18next';
import { AudioTrimmer, DEFAULT_TRIM_PRESETS } from '@/components/audio-trimmer/audio-trimmer';
import type { TrimRange } from '@/components/audio-trimmer/trim-math';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { REF_TEXT_MAX_SECONDS } from '@/lib/api/generate';
import type { Profile } from '@/lib/api/types';
import type { ReferenceOrigin } from '@/lib/store/reference';

/** Mono decode rate of the exported reference: the rate the cloning engines run at. */
export const REFERENCE_DECODE_RATE = 24000;
/** The trimmer opens on this much of the clip, from where speech starts. */
export const REFERENCE_DEFAULT_WINDOW_SECONDS = 15;

/** What the reference trimmer cuts from: a picked file, or a saved voice's stored clip. */
export interface ReferenceTrimSource {
  source: File | string;
  name: string;
  /** The last cut, so reopening lands on it with the whole recording around it. */
  initialRange?: TrimRange | null;
}

/** A trimmer session: what is being cut and whether the flow demanded it. */
export interface ReferenceTrimSession extends ReferenceTrimSource {
  /** Opened by itself because the clip was over the limit; cancelling drops it. */
  required: boolean;
}

/** The session to reopen an accepted reference: its full original when it came from a cut. */
export function reopenSession(
  file: File,
  origin: ReferenceOrigin | null | undefined,
): ReferenceTrimSession {
  return origin
    ? {
        source: origin.source,
        name: origin.name,
        initialRange: { start: origin.start, end: origin.end },
        required: false,
      }
    : { source: file, name: file.name, initialRange: null, required: false };
}

/** The stored reference as a backend path the trimmer can fetch (never the /api-prefixed player URL). */
export function profileReferencePath(profile: Pick<Profile, 'id' | 'audio_url'>): string {
  return profile.audio_url?.startsWith('/profiles/')
    ? profile.audio_url
    : `/profiles/${encodeURIComponent(profile.id)}/audio`;
}

/**
 * Pick the stretch of a clip the voice learns from. Opens by itself for clips
 * over REF_TEXT_MAX_SECONDS and on request for shorter ones; reopening a cut
 * shows the whole recording with the last cut preselected. Whatever is
 * confirmed here is the reference: the original never leaves this dialog.
 */
export function ReferenceTrimDialog({
  source,
  name,
  initialRange = null,
  onTrimmed,
  onCancel,
}: ReferenceTrimSource & {
  onTrimmed: (file: File, durationSeconds: number, range: TrimRange) => void;
  onCancel: () => void;
}) {
  const { t } = useTranslation();
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onCancel();
      }}
    >
      <DialogContent showCloseButton={false} className="min-w-0 sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>{t('referenceTrim.title')}</DialogTitle>
          <DialogDescription>
            {t('referenceTrim.description')}{' '}
            {t('referenceTrim.limit', { max: REF_TEXT_MAX_SECONDS })}
          </DialogDescription>
        </DialogHeader>
        <AudioTrimmer
          source={source}
          name={name}
          maxSeconds={REF_TEXT_MAX_SECONDS}
          presets={DEFAULT_TRIM_PRESETS}
          defaultLength={REFERENCE_DEFAULT_WINDOW_SECONDS}
          initialRange={initialRange}
          decodeSampleRate={REFERENCE_DECODE_RATE}
          zoom
          confirmLabel={t('referenceTrim.use')}
          onSave={(file, range) => onTrimmed(file, range.end - range.start, range)}
          onCancel={onCancel}
        />
      </DialogContent>
    </Dialog>
  );
}
