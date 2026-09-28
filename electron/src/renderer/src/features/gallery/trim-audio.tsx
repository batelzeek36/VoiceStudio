import { useTranslation } from 'react-i18next';
import { ArrowLeftIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { AudioTrimmer } from '@/components/audio-trimmer/audio-trimmer';
import { DEFAULT_MAX_TRIM_SECONDS, trimmedAudio } from '@/components/audio-trimmer/trimmed-audio';

/** The export helper stays importable from here for the gallery's tests. */
export { trimmedAudio };

const MAX_TRIM_SECONDS = DEFAULT_MAX_TRIM_SECONDS;

/**
 * The gallery's full-page frame around the shared trimmer: a back arrow, the
 * import's name, and a 60 s ceiling. The saved file keeps the import's name so
 * the new gallery entry is called "<name> - Trim".
 */
export function TrimAudio({
  src,
  name,
  busy,
  saveError,
  onSave,
  onCancel,
}: {
  src: string;
  name: string;
  busy: boolean;
  saveError?: string;
  onSave(file: File): void;
  onCancel(): void;
}) {
  const { t } = useTranslation();
  return (
    <section className="min-h-0 flex-1 overflow-y-auto p-6">
      <div className="mx-auto max-w-5xl space-y-5">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon-sm"
            disabled={busy}
            aria-label={t('common.cancel')}
            onClick={onCancel}
          >
            <ArrowLeftIcon />
          </Button>
          <div>
            <h2 className="text-sm font-medium">{t('trimmer.title')}</h2>
            <p className="text-xs text-muted-foreground">{name}</p>
          </div>
        </div>
        <AudioTrimmer
          source={src}
          name={name}
          maxSeconds={MAX_TRIM_SECONDS}
          defaultLength={MAX_TRIM_SECONDS}
          busy={busy}
          saveError={saveError}
          playerSource="gallery-trim"
          onSave={(file) => onSave(new File([file], name + '.wav', { type: 'audio/wav' }))}
          onCancel={onCancel}
        />
      </div>
    </section>
  );
}
