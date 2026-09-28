import { encodeWav, sliceToMono } from '../../../../../../frontend/src/utils/audioTrim';

/** The gallery's ceiling; the clone reference trimmer passes its own. */
export const DEFAULT_MAX_TRIM_SECONDS = 60;

/**
 * Cut [start, end] out of a decoded buffer as a mono 16-bit PCM WAV at the
 * buffer's sample rate. Rejects anything outside the clip, shorter than 20 ms
 * or longer than `maxSeconds`, so a caller never exports what the UI refused.
 */
export function trimmedAudio(
  buffer: AudioBuffer,
  start: number,
  end: number,
  maxSeconds = DEFAULT_MAX_TRIM_SECONDS,
): Blob {
  if (
    !Number.isFinite(start) ||
    !Number.isFinite(end) ||
    start < 0 ||
    end > buffer.duration ||
    end - start < 0.02 ||
    end - start > maxSeconds + 0.001
  )
    throw new Error('Invalid trim range');
  return new Blob([encodeWav(sliceToMono(buffer, start, end), buffer.sampleRate)], {
    type: 'audio/wav',
  });
}

/** "Take 3 4.2-16.8s.wav": the source name without its extension plus the cut. */
export function trimmedFileName(sourceName: string, start: number, end: number): string {
  const stem = sourceName.replace(/\.[A-Za-z0-9]{1,5}$/, '').trim() || 'reference';
  return `${stem} ${start.toFixed(1)}-${end.toFixed(1)}s.wav`;
}
