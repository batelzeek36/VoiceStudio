import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  act,
  cleanup,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor,
} from '@testing-library/react';
import type { ReactNode } from 'react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { Profile } from '@/lib/api/types';
import { cloneSettingsStore, patchCloneSettings } from '@/lib/store/clone-settings';
import { clearReference, referenceStore, useReference } from '@/lib/store/reference';
import { useReferenceTranscript } from '@/hooks/use-reference-transcript';

/**
 * The trimmer's place in the new-voice flow: an over-long pick opens it, what
 * it confirms replaces the reference (never the original), the transcript is
 * taken from that trimmed clip, Trim reopens the whole original on the last
 * cut, and cancelling a demanded trim says so. The dialog itself is stubbed:
 * its decoding and playback are exercised in the browser smoke, the math in
 * trim-math.test.ts.
 */
const original = new File(['a'.repeat(4000)], 'long take.m4a', { type: 'audio/mp4' });
const trimmed = new File(['cut'], 'long take 4.2-16.8s.wav', { type: 'audio/wav' });
const cut = { start: 4.2, end: 16.8 };

const mock = vi.hoisted(() => ({
  dialogs: [] as Array<{ source: File | string; name: string; initialRange: unknown }>,
  replace: vi.fn(),
  json: vi.fn(),
  toast: Object.assign(vi.fn(), { error: vi.fn(), success: vi.fn(), warning: vi.fn() }),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key, i18n: { language: 'en' } }),
}));
vi.mock('sonner', () => ({ toast: mock.toast }));
vi.mock('@tanstack/react-router', () => ({
  Link: ({ children }: { children: ReactNode }) => <a href="#models">{children}</a>,
}));
vi.mock('@/lib/store/workspace', () => ({ setWorkspace: vi.fn() }));
vi.mock('@/hooks/use-engines', () => ({ useEngines: () => ({ activeTts: null }) }));
vi.mock('@/hooks/use-profiles', () => ({
  useCreateCloneProfile: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useProfiles: () => ({ data: [] }),
  useDeleteProfile: () => ({ isPending: false, mutateAsync: vi.fn() }),
  useReplaceProfileAudio: () => ({ mutateAsync: mock.replace }),
}));
vi.mock('@/hooks/use-tts-readiness', () => ({ useTtsReadiness: () => null }));
vi.mock('@/lib/audio/object-url', () => ({
  createObjectUrl: (blob: Blob) => 'blob:' + (blob instanceof File ? blob.name : 'clip'),
  revokeObjectUrl: vi.fn(),
}));
vi.mock('@/lib/api/client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/api/client')>()),
  apiJson: mock.json,
}));
vi.mock('@/components/waveform-player', () => ({
  WaveformPlayer: ({ src }: { src: string }) => <div data-testid="player">{src}</div>,
}));
vi.mock('./portrait-search', () => ({ PortraitSearch: () => null }));
vi.mock('./profile-image-editor', () => ({ ProfileImageEditor: () => null }));
vi.mock('./profile-consent', () => ({ ProfileConsent: () => null }));
vi.mock('./profile-preview', () => ({ ProfilePreview: () => null }));
vi.mock('./profile-usage', () => ({ ProfileUsagePanel: () => null }));
vi.mock('./language-picker', () => ({ LanguagePicker: () => null }));
vi.mock('./persona-export', () => ({ PersonaExport: () => null }));
// The zones expose the two ways a pick leaves them: accepted, or sent to trim.
vi.mock('./reference-input', () => ({
  UploadZone: ({
    onAccept,
    onTrim,
  }: {
    onAccept?: (file: File, seconds: number | null) => void;
    onTrim: (file: File, seconds: number) => void;
  }) => (
    <>
      <button type="button" onClick={() => onTrim(original, 35.8)}>
        pick-long
      </button>
      <button
        type="button"
        onClick={() => onAccept?.(new File(['short'], 'short.wav', { type: 'audio/wav' }), 9)}
      >
        pick-short
      </button>
    </>
  ),
  RecordZone: () => null,
  ReferenceUsageNote: () => null,
}));
vi.mock('./reference-trim', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./reference-trim')>()),
  ReferenceTrimDialog: ({
    source,
    name,
    initialRange,
    onTrimmed,
    onCancel,
  }: {
    source: File | string;
    name: string;
    initialRange?: { start: number; end: number } | null;
    onTrimmed: (file: File, seconds: number, range: { start: number; end: number }) => void;
    onCancel: () => void;
  }) => {
    mock.dialogs.push({ source, name, initialRange: initialRange ?? null });
    return (
      <div role="dialog" aria-label="trim-dialog">
        <button type="button" onClick={() => onTrimmed(trimmed, 12.6, cut)}>
          confirm-trim
        </button>
        <button type="button" onClick={onCancel}>
          cancel-trim
        </button>
      </div>
    );
  },
}));
import { ReferencePanel } from './reference-panel';
import { EditProfile } from './edit-profile';

beforeEach(() => {
  mock.json.mockReset();
  mock.json.mockResolvedValue({ text: 'trimmed words' });
});
afterEach(() => {
  cleanup();
  clearReference();
  patchCloneSettings({ selectedProfileId: null, refText: '', instruct: '' });
  mock.dialogs.length = 0;
  mock.replace.mockReset();
  mock.toast.mockClear();
});

it('opens the trimmer for an over-long pick and makes only the trimmed clip the reference', async () => {
  render(<ReferencePanel />);
  expect(screen.queryByRole('dialog')).toBeNull();

  fireEvent.click(screen.getByText('pick-long'));

  expect(screen.getByRole('dialog', { name: 'trim-dialog' })).toBeInTheDocument();
  expect(mock.dialogs.at(-1)).toEqual({
    source: original,
    name: 'long take.m4a',
    initialRange: null,
  });
  // Nothing is the reference until the trimmer confirms.
  expect(referenceStore.state.file).toBeNull();

  fireEvent.click(screen.getByText('confirm-trim'));

  await waitFor(() => expect(referenceStore.state.file).toBe(trimmed));
  expect(referenceStore.state.durationSeconds).toBe(12.6);
  expect(referenceStore.state.origin).toEqual({
    source: original,
    name: 'long take.m4a',
    ...cut,
  });
  expect(screen.queryByRole('dialog')).toBeNull();
  expect(screen.getByText('long take 4.2-16.8s.wav')).toBeInTheDocument();
  expect(mock.toast).not.toHaveBeenCalled();
});

it('reopens the whole original on the last cut when a trimmed reference is trimmed again', async () => {
  render(<ReferencePanel />);
  fireEvent.click(screen.getByText('pick-long'));
  fireEvent.click(screen.getByText('confirm-trim'));
  await waitFor(() => expect(referenceStore.state.file).toBe(trimmed));

  fireEvent.click(screen.getByRole('button', { name: /referenceTrim.trim/ }));

  expect(mock.dialogs.at(-1)).toEqual({
    source: original,
    name: 'long take.m4a',
    initialRange: cut,
  });
  // Cancelling a re-trim keeps the cut that was already accepted, quietly.
  fireEvent.click(screen.getByText('cancel-trim'));
  expect(referenceStore.state.file).toBe(trimmed);
  expect(mock.toast).not.toHaveBeenCalled();
});

it('says why an over-long pick was not kept when its trim is cancelled', () => {
  render(<ReferencePanel />);
  fireEvent.click(screen.getByText('pick-long'));
  fireEvent.click(screen.getByText('cancel-trim'));
  expect(screen.queryByRole('dialog')).toBeNull();
  expect(referenceStore.state.file).toBeNull();
  expect(mock.toast).toHaveBeenCalledWith('referenceTrim.not_kept');
});

it('offers Trim on an accepted short clip and swaps in the cut', async () => {
  const short = new File(['short'], 'short.wav', { type: 'audio/wav' });
  await act(async () => {
    const { setReferenceFile } = await import('@/lib/store/reference');
    await setReferenceFile(short, 9);
  });
  render(<ReferencePanel />);
  expect(screen.queryByRole('dialog')).toBeNull();

  fireEvent.click(screen.getByRole('button', { name: /referenceTrim.trim/ }));

  expect(mock.dialogs.at(-1)).toEqual({ source: short, name: 'short.wav', initialRange: null });
  fireEvent.click(screen.getByText('confirm-trim'));
  await waitFor(() => expect(referenceStore.state.file).toBe(trimmed));
  expect(referenceStore.state.origin?.source).toBe(short);
});

it('transcribes the trimmed clip, never the original', async () => {
  render(<ReferencePanel />);
  fireEvent.click(screen.getByText('pick-long'));
  fireEvent.click(screen.getByText('confirm-trim'));
  await waitFor(() => expect(referenceStore.state.file).toBe(trimmed));

  // The clone page feeds the store's reference straight into the transcript hook.
  const { result } = renderHook(() => useReferenceTranscript(useReference().file));
  await waitFor(() => expect(result.current.state).toBe('ready'));

  expect(mock.json).toHaveBeenCalledTimes(1);
  const body = mock.json.mock.calls[0][1]?.body as FormData;
  expect(body.get('audio')).toBe(trimmed);
  expect(body.get('audio')).not.toBe(original);
  expect(cloneSettingsStore.state.refText).toBe('trimmed words');
});

const profile = {
  id: 'v1',
  name: 'Scarlet',
  kind: 'clone',
  ref_audio_path: 'v1.wav',
  audio_url: '/profiles/v1/audio?v=1',
  ref_text: 'old words',
  instruct: '',
  language: 'Auto',
  created_at: 1,
  is_locked: 1,
} as unknown as Profile;

function renderEditor(onDone = vi.fn()) {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <EditProfile profile={profile} onDone={onDone} />
    </QueryClientProvider>,
  );
  return onDone;
}

it('trims an over-long replacement pick and saves the cut with a fresh transcript', async () => {
  mock.replace.mockResolvedValue({ ...profile });
  const onDone = renderEditor();

  fireEvent.click(screen.getByRole('button', { name: /clone.replace_reference/ }));
  fireEvent.click(screen.getByText('pick-long'));
  expect(mock.dialogs.at(-1)).toEqual({
    source: original,
    name: 'long take.m4a',
    initialRange: null,
  });
  // The over-long original is never shown as the replacement.
  expect(screen.queryByText(/long take.m4a/)).toBeNull();

  fireEvent.click(screen.getByText('confirm-trim'));

  expect(screen.getByTestId('player')).toHaveTextContent('blob:long take 4.2-16.8s.wav');
  expect(screen.getByLabelText('clone.transcript')).toHaveValue('');
  fireEvent.submit(screen.getByRole('button', { name: 'clone.save' }).closest('form')!);
  await waitFor(() => expect(onDone).toHaveBeenCalledOnce());
  expect(mock.replace).toHaveBeenCalledWith(
    expect.objectContaining({
      id: 'v1',
      refAudio: trimmed,
      refAudioName: trimmed.name,
      refText: '',
    }),
  );
});

it('reopens a trimmed replacement on its whole original and toasts a cancelled demanded trim', () => {
  renderEditor();
  fireEvent.click(screen.getByRole('button', { name: /clone.replace_reference/ }));
  fireEvent.click(screen.getByText('pick-long'));
  fireEvent.click(screen.getByText('cancel-trim'));
  expect(mock.toast).toHaveBeenCalledWith('referenceTrim.not_kept');
  expect(screen.queryByTestId('player')).toHaveTextContent('/profiles/v1/audio?v=1');

  fireEvent.click(screen.getByText('pick-long'));
  fireEvent.click(screen.getByText('confirm-trim'));
  expect(screen.getByTestId('player')).toHaveTextContent('blob:long take 4.2-16.8s.wav');

  fireEvent.click(screen.getByRole('button', { name: /referenceTrim.trim/ }));
  expect(mock.dialogs.at(-1)).toEqual({
    source: original,
    name: 'long take.m4a',
    initialRange: cut,
  });
});

it('trims the stored reference from its versioned backend path', () => {
  renderEditor();

  fireEvent.click(screen.getByRole('button', { name: /referenceTrim.trim/ }));

  expect(mock.dialogs.at(-1)).toEqual({
    source: '/profiles/v1/audio?v=1',
    name: 'Scarlet',
    initialRange: null,
  });
  fireEvent.click(screen.getByText('confirm-trim'));
  expect(screen.getByTestId('player')).toHaveTextContent('blob:long take 4.2-16.8s.wav');
  // Keep current reference still restores the transcript from before the trim.
  fireEvent.click(screen.getByRole('button', { name: /clone.keep_reference/ }));
  expect(screen.getByLabelText('clone.transcript')).toHaveValue('old words');
  expect(screen.getByTestId('player')).toHaveTextContent('/profiles/v1/audio?v=1');
});

it('offers Trim on an accepted replacement and swaps in the cut', () => {
  renderEditor();
  fireEvent.click(screen.getByRole('button', { name: /clone.replace_reference/ }));
  fireEvent.click(screen.getByText('pick-short'));
  expect(screen.getByTestId('player')).toHaveTextContent('blob:short.wav');

  fireEvent.click(screen.getByRole('button', { name: /referenceTrim.trim/ }));
  expect(mock.dialogs.at(-1)?.name).toBe('short.wav');
  expect(mock.dialogs.at(-1)?.initialRange).toBeNull();
  fireEvent.click(screen.getByText('confirm-trim'));
  expect(screen.getByTestId('player')).toHaveTextContent('blob:long take 4.2-16.8s.wav');
});
