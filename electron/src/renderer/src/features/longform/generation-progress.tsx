import { useEffect, useMemo, useRef, useState } from 'react';
import { CheckIcon, CircleIcon, LoaderCircleIcon, XIcon, ZapIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import {
  liveRemaining,
  longformRemaining,
  renderedChapterEta,
  type LiveCountdown,
} from '@/lib/render-estimate-time';
import type { AudiobookRenderChapter } from './longform-session';

function formatElapsed(seconds: number): string {
  const safe = Math.max(0, Math.floor(seconds));
  return `${Math.floor(safe / 60)}:${String(safe % 60).padStart(2, '0')}`;
}

function ChapterStatusIcon({ status }: { status: string }) {
  if (status === 'rendering')
    return <LoaderCircleIcon className="size-3.5 animate-spin motion-reduce:animate-none" />;
  if (status === 'done') return <CheckIcon className="size-3.5" />;
  if (status === 'cached') return <ZapIcon className="size-3.5" />;
  if (status === 'failed') return <XIcon className="size-3.5" />;
  return <CircleIcon className="size-3.5" />;
}

export function GenerationProgress({
  chapters,
  assembling,
  planned = null,
  live = null,
}: {
  chapters: AudiobookRenderChapter[];
  assembling: boolean;
  /** Estimated seconds per chapter on this machine (render-time estimate). */
  planned?: (number | null)[] | null;
  /** The backend's per-call countdown; preferred over the per-chapter plan. */
  live?: LiveCountdown | null;
}) {
  const { t } = useTranslation();
  const start = useRef(performance.now());
  const [now, setNow] = useState(start.current);
  const [lastChapterAt, setLastChapterAt] = useState(start.current);
  const completed = useMemo(
    () =>
      chapters.filter((chapter) => ['done', 'cached', 'failed'].includes(chapter.status)).length,
    [chapters],
  );
  const total = chapters.length;
  const percent = total ? Math.round((completed / total) * 100) : 0;
  const elapsed = (now - start.current) / 1000;
  const statuses = chapters.map((chapter) => chapter.status);
  const open = total - completed;
  // The backend's countdown, re-fitted after every engine call; else the plan
  // re-fitted as chapters finish; without either (first renders on this
  // machine, a resume) the pace of the chapters actually rendered.
  const eta = assembling
    ? null
    : live
      ? liveRemaining(live, now)
      : (longformRemaining(planned, statuses, elapsed, Math.max(0, (now - lastChapterAt) / 1000)) ??
        renderedChapterEta(statuses, elapsed));

  useEffect(() => {
    const timer = window.setInterval(() => setNow(performance.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    setLastChapterAt(performance.now());
  }, [completed]);

  return (
    <div
      role="status"
      aria-live="polite"
      className="rounded-xl border border-primary/15 bg-primary/5 p-3 shadow-[inset_0_1px_0_rgb(255_255_255/4%)]"
    >
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="font-medium">
          {assembling
            ? t('audiobook.assembling')
            : t('audiobook.progress_summary', { current: completed, total })}
        </span>
        <span className="shrink-0 font-mono tabular-nums text-muted-foreground">
          {formatElapsed(elapsed)}
          {eta == null
            ? ''
            : eta >= 1
              ? ` · ${t('audiobook.eta', { time: formatElapsed(eta) })}`
              : open <= 1
                ? ` · ${t('renderEstimate.finishing')}`
                : ''}
        </span>
      </div>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-300 motion-reduce:transition-none"
          style={{ width: `${assembling ? 100 : percent}%` }}
        />
      </div>
      {total > 0 && (
        <ol className="mt-2 max-h-36 space-y-0.5 overflow-y-auto">
          {chapters.map((chapter, index) => (
            <li
              key={index}
              title={chapter.status === 'failed' ? chapter.error : undefined}
              className="flex min-w-0 items-center gap-2 py-0.5 text-xs"
            >
              <span
                className={
                  chapter.status === 'failed'
                    ? 'text-destructive'
                    : chapter.status === 'rendering'
                      ? 'text-primary'
                      : 'text-muted-foreground'
                }
              >
                <ChapterStatusIcon status={chapter.status} />
              </span>
              <span
                className={`truncate ${
                  chapter.status === 'pending'
                    ? 'text-muted-foreground/55'
                    : chapter.status === 'rendering'
                      ? 'font-medium'
                      : 'text-muted-foreground'
                }`}
              >
                {chapter.title || t('audiobook.chapter_n', { n: index + 1 })}
              </span>
              {chapter.status === 'cached' && (
                <span className="shrink-0 text-muted-foreground">
                  · {t('audiobook.cached_tag')}
                </span>
              )}
              {chapter.status === 'failed' && (
                <span className="min-w-0 truncate text-destructive/80">
                  · {t('audiobook.failed_tag')}
                  {chapter.error ? `: ${chapter.error}` : ''}
                </span>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
