import { cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

const mock = vi.hoisted(() => ({
  probes: new Map<string, (seconds: number) => void>(),
  toastWarning: vi.fn(),
  toastError: vi.fn(),
  setReferenceFile: vi.fn(async (_file: File, _durationSeconds?: number | null) => ({
    ok: true,
    durationSeconds: 9,
    tooLong: false,
  })),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('sonner', () => ({ toast: { error: mock.toastError, warning: mock.toastWarning } }));
vi.mock('@/hooks/use-engines', () => ({ useEngines: () => ({ activeTts: null }) }));
vi.mock('@/lib/audio/probe', () => ({
  probeAudioDuration: (file: File) =>
    new Promise<number>((resolve) => mock.probes.set(file.name, resolve)),
}));
vi.mock('@/lib/store/reference', () => ({ setReferenceFile: mock.setReferenceFile }));
vi.mock('@/hooks/use-recording', () => ({ useRecording: vi.fn() }));
vi.mock('@/components/recording-inputs', () => ({ RecordingInputs: () => null }));
import { UploadZone } from './reference-input';

afterEach(() => {
  cleanup();
  mock.probes.clear();
  vi.clearAllMocks();
});

const clip = (name: string) => new File(['audio'], name, { type: 'audio/wav' });

it('keeps the latest pick when an earlier clip finishes probing last', async () => {
  const onAccept = vi.fn();
  const onTrim = vi.fn();
  const { container } = render(<UploadZone onAccept={onAccept} onTrim={onTrim} />);
  const input = container.querySelector('input[type="file"]')!;

  fireEvent.change(input, { target: { files: [clip('first.wav')] } });
  fireEvent.change(input, { target: { files: [clip('second.wav')] } });
  mock.probes.get('second.wav')!(5);
  await vi.waitFor(() => expect(onAccept).toHaveBeenCalledOnce());
  mock.probes.get('first.wav')!(90);
  await new Promise((resolve) => setTimeout(resolve, 0));

  expect(onAccept).toHaveBeenCalledOnce();
  expect(onAccept.mock.calls[0][0].name).toBe('second.wav');
  // The stale clip neither opens the trimmer nor toasts over the kept clip.
  expect(onTrim).not.toHaveBeenCalled();
  expect(mock.toastError).not.toHaveBeenCalled();
  expect(mock.toastWarning).not.toHaveBeenCalled();
});

it('sends a clip over the transcript limit to the trimmer instead of accepting it', async () => {
  const onAccept = vi.fn();
  const onTrim = vi.fn();
  const { container } = render(<UploadZone onAccept={onAccept} onTrim={onTrim} />);
  fireEvent.change(container.querySelector('input[type="file"]')!, {
    target: { files: [clip('long.wav')] },
  });
  mock.probes.get('long.wav')!(40);
  await vi.waitFor(() => expect(onTrim).toHaveBeenCalledOnce());
  expect(onTrim.mock.calls[0][0].name).toBe('long.wav');
  expect(onTrim.mock.calls[0][1]).toBe(40);
  expect(onAccept).not.toHaveBeenCalled();
  expect(mock.toastWarning).not.toHaveBeenCalled();
  expect(mock.toastError).not.toHaveBeenCalled();
});

it('accepts a clip within the limit without opening the trimmer', async () => {
  const onAccept = vi.fn();
  const onTrim = vi.fn();
  const { container } = render(<UploadZone onAccept={onAccept} onTrim={onTrim} />);
  fireEvent.change(container.querySelector('input[type="file"]')!, {
    target: { files: [clip('fits.wav')] },
  });
  mock.probes.get('fits.wav')!(20);
  await vi.waitFor(() => expect(onAccept).toHaveBeenCalledOnce());
  expect(onAccept.mock.calls[0][1]).toBe(20);
  expect(onTrim).not.toHaveBeenCalled();
});

it('hands the composer an accepted clip with its measured length', async () => {
  const onTrim = vi.fn();
  const { container } = render(<UploadZone onTrim={onTrim} />);
  fireEvent.change(container.querySelector('input[type="file"]')!, {
    target: { files: [clip('short.wav')] },
  });
  mock.probes.get('short.wav')!(9);
  await vi.waitFor(() => expect(mock.setReferenceFile).toHaveBeenCalledOnce());
  expect(mock.setReferenceFile.mock.calls[0][0].name).toBe('short.wav');
  expect(mock.setReferenceFile.mock.calls[0][1]).toBe(9);
  expect(onTrim).not.toHaveBeenCalled();
});

it('refuses a recording too long to decode here', async () => {
  const onTrim = vi.fn();
  const { container } = render(<UploadZone onTrim={onTrim} />);
  fireEvent.change(container.querySelector('input[type="file"]')!, {
    target: { files: [clip('podcast.wav')] },
  });
  mock.probes.get('podcast.wav')!(45 * 60);
  await vi.waitFor(() => expect(mock.toastError).toHaveBeenCalledOnce());
  expect(mock.toastError.mock.calls[0][0]).toBe('referenceTrim.input_too_long');
  expect(onTrim).not.toHaveBeenCalled();
  expect(mock.setReferenceFile).not.toHaveBeenCalled();
});
