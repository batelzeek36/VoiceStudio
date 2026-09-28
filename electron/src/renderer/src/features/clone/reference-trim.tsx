import { useTranslation } from 'react-i18next';
import { AudioTrimmer, DEFAULT_TRIM_PRESETS } from '@/components/audio-trimmer/audio-trimmer';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { REF_TEXT_MAX_SECONDS } from '@/lib/api/generate';
import type { Profile } from '@/lib/api/types';

/** Mono decode rate of the exported reference: the rate the cloning engines run at. */
export const REFERENCE_DECODE_RATE = 24000;
/** The trimmer opens on this much of the clip, from where speech starts. */
export const REFERENCE_DEFAULT_WINDOW_SECONDS = 15;

/** What the reference trimmer cuts from: a picked file, or a saved voice's stored clip. */
export interface ReferenceTrimSource {
  source: File | string;
  name: string;
}

/** The stored reference as a backend path the trimmer can fetch (never the /api-prefixed player URL). */
export function profileReferencePath(profile: Pick<Profile, 'id' | 'audio_url'>): string {
  return profile.audio_url?.startsWith('/profiles/')
    ? profile.audio_url
    : `/profiles/${encodeURIComponent(profile.id)}/audio`;
}

/**
 * Pick the stretch of a clip the voice learns from. Opens by itself for clips
 * over REF_TEXT_MAX_SECONDS (the engine aligns a transcript against the whole
 * clip, so longer ones are refused) and on request for shorter ones. Whatever
 * is confirmed here is the reference: the original never leaves this dialog.
 */
export function ReferenceTrimDialog({
  source,
  name,
  onTrimmed,
  onCancel,
}: ReferenceTrimSource & {
  onTrimmed: (file: File, durationSeconds: number) => void;
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
      <DialogContent showCloseButton={false} className="sm:max-w-3xl">
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
          decodeSampleRate={REFERENCE_DECODE_RATE}
          playerSource="reference-trim"
          confirmLabel={t('referenceTrim.use')}
          onSave={(file, range) => onTrimmed(file, range.end - range.start)}
          onCancel={onCancel}
        />
      </DialogContent>
    </Dialog>
  );
}
