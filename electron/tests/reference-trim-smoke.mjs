/**
 * Reference trimmer smoke: a real over-long clip dropped on Voice Clone opens
 * the trimmer; presets, keys, drag, snap, loop, the 20 s cap and the confirm
 * are driven for real, and the transcript request is checked to carry the
 * trimmed WAV rather than the original file.
 *
 * Self-contained: starts the fake health backend and the renderer smoke server
 * on VOICESTUDIO_SMOKE_PORT (default 3915), never the live backend ports.
 *
 *   VOICESTUDIO_TRIM_CLIP=/path/to/clip-over-20s.m4a \
 *   VOICESTUDIO_TRIM_SHOTS=/tmp/shots node tests/reference-trim-smoke.mjs
 *
 * PLAYWRIGHT_CHANNEL defaults to `chrome` so AAC (m4a) decodes as it does in
 * Electron's Chromium; the bundled open-source Chromium has no AAC decoder.
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdirSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import assert from 'node:assert/strict';
import { startHealthBackend } from './fake-health-backend.mjs';

const clip = process.env.VOICESTUDIO_TRIM_CLIP;
assert(clip, 'VOICESTUDIO_TRIM_CLIP must point at an audio file longer than 20 s');
const shots = process.env.VOICESTUDIO_TRIM_SHOTS || join(tmpdir(), 'voicestudio-reference-trim');
mkdirSync(shots, { recursive: true });
const uiPort = Number(process.env.VOICESTUDIO_SMOKE_PORT) || 3915;
assert(![3900, 3901, 9333].includes(uiPort), 'never the live backend ports');
const REFERENCE_RATE = 24000;

const backend = await startHealthBackend();
const server = spawn('bunx', ['vite', '--config', 'tests/renderer-smoke.vite.config.mjs'], {
  cwd: resolve(import.meta.dirname, '..'),
  env: {
    ...process.env,
    OMNIVOICE_PORT: String(backend.port),
    VOICESTUDIO_SMOKE_PORT: String(uiPort),
  },
  stdio: ['ignore', 'pipe', 'pipe'],
});
const serverLog = [];
server.stdout.on('data', (chunk) => serverLog.push(String(chunk)));
server.stderr.on('data', (chunk) => serverLog.push(String(chunk)));
const origin = `http://localhost:${uiPort}`;
for (let attempt = 0; ; attempt++) {
  try {
    const response = await fetch(origin + '/');
    if (response.ok) break;
  } catch {
    /* not up yet */
  }
  assert(attempt < 120, 'renderer smoke server did not start:\n' + serverLog.join(''));
  await new Promise((done) => setTimeout(done, 500));
}

const engines = {
  tts: {
    active: 'omnivoice',
    backends: [
      {
        id: 'omnivoice',
        name: 'OmniVoice',
        available: true,
        supports_cloning: true,
        max_ref_seconds: 15,
        ref_strategy: 'best_window',
      },
    ],
  },
  asr: { active: 'whisper', backends: [{ id: 'whisper', name: 'Whisper', available: true }] },
  llm: { active: null, backends: [] },
};

const browser = await chromium.launch({
  channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
  headless: true,
});
const readoutPattern = /(\d+:\d\d\.\d) – (\d+:\d\d\.\d) · (\d+\.\d) s/;
const toSeconds = (clock) => {
  const [minutes, seconds] = clock.split(':');
  return Number(minutes) * 60 + Number(seconds);
};
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
  await page.addInitScript(() => {
    localStorage.setItem('voicestudio.setup.complete.v1', '1');
    localStorage.setItem('voicestudio.locale', 'en');
  });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const transcribeBodies = [];
  const apiLog = [];
  page.on('request', (request) => {
    if (request.url().includes('/api/')) apiLog.push(request.method() + ' ' + request.url());
  });
  await page.route('**/api/engines', (route) => route.fulfill({ json: engines }));
  await page.route('**/api/profiles', (route) => route.fulfill({ json: [] }));
  await page.route('**/api/transcribe', (route) => {
    transcribeBodies.push(route.request().postDataBuffer());
    return route.fulfill({ json: { text: 'and yes dear I am using myself as the example' } });
  });
  await page.goto(origin + '/#/clone');
  await page.getByText('Drop audio here', { exact: false }).waitFor({ timeout: 30000 });

  // 1. An over-long pick opens the trimmer by itself.
  await page.locator('input[type="file"]').first().setInputFiles(clip);
  const dialog = page.getByRole('dialog');
  await dialog.waitFor({ timeout: 20000 });
  await dialog.getByRole('button', { name: 'Use this passage', exact: true }).waitFor({
    timeout: 30000,
  });
  await dialog.locator('[part~="region"]').waitFor();
  const readout = dialog.getByText(readoutPattern);
  const read = async () => {
    const match = (await readout.innerText()).match(readoutPattern);
    assert(match, 'readout missing');
    return { start: toSeconds(match[1]), end: toSeconds(match[2]), length: Number(match[3]) };
  };
  let selection = await read();
  assert(selection.length <= 15 && selection.length >= 14.6, `default window ${selection.length}`);
  await page.screenshot({ path: join(shots, 'reference-trim-open.png') });

  // 2. Presets keep the start and set the length; 1/2/3 do the same.
  const startBefore = selection.start;
  await dialog.getByRole('button', { name: /^10 s/ }).click();
  selection = await read();
  assert(selection.length <= 10 && selection.length >= 9.6, `10 s preset ${selection.length}`);
  assert(Math.abs(selection.start - startBefore) <= 0.35, 'preset moved the start');
  await page.keyboard.press('3');
  selection = await read();
  assert(selection.length <= 20 && selection.length >= 19.6, `20 s key ${selection.length}`);
  await page.keyboard.press('2');
  selection = await read();
  assert(selection.length <= 15 && selection.length >= 14.6, `15 s key ${selection.length}`);
  await page.keyboard.press('1');
  selection = await read();
  assert(selection.length <= 10 && selection.length >= 9.6, `10 s key ${selection.length}`);

  // 3. Dragging on empty waveform draws a new selection; a long drag is capped.
  const wave = dialog.locator('[data-slot="audio-trimmer-wave"]');
  const box = await wave.boundingBox();
  assert(box);
  const drag = async (fromFraction, toFraction) => {
    const y = box.y + box.height / 2;
    await page.mouse.move(box.x + box.width * fromFraction, y);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * (fromFraction + 0.02), y, { steps: 3 });
    await page.mouse.move(box.x + box.width * toFraction, y, { steps: 12 });
    await page.mouse.up();
  };
  await drag(0.6, 0.8);
  selection = await read();
  assert(selection.start > 18, `drag start ${selection.start}`);
  assert(selection.length >= 5 && selection.length <= 8.5, `drag length ${selection.length}`);
  await drag(0.05, 0.95);
  selection = await read();
  assert(selection.length <= 20, `capped drag ${selection.length}`);
  await page.keyboard.press('1');

  // 4. Snap off makes a preset exact; on again snaps it into a pause.
  const snap = dialog.getByRole('switch', { name: 'Snap to silence' });
  assert.equal(await snap.getAttribute('aria-checked'), 'true');
  await snap.click();
  assert.equal(await snap.getAttribute('aria-checked'), 'false');
  await page.keyboard.press('2');
  selection = await read();
  assert.equal(selection.length, 15, `unsnapped preset ${selection.length}`);
  await snap.click();
  await page.keyboard.press('2');
  selection = await read();
  assert(selection.length <= 15, `snapped preset ${selection.length}`);

  // 5. Loop toggles; the preview plays.
  const loop = dialog.getByRole('button', { name: 'Loop preview', exact: true });
  assert.equal(await loop.getAttribute('aria-pressed'), 'true');
  await loop.click();
  assert.equal(await loop.getAttribute('aria-pressed'), 'false');
  await loop.click();
  await dialog.getByRole('button', { name: 'Play', exact: true }).click();
  await dialog.getByRole('button', { name: 'Pause', exact: true }).waitFor({ timeout: 10000 });
  await dialog.getByRole('button', { name: 'Pause', exact: true }).click();

  // 6. The cap holds when the end is typed past it.
  const startField = dialog.getByRole('spinbutton', { name: 'Start', exact: true });
  const endField = dialog.getByRole('spinbutton', { name: 'End', exact: true });
  const start = Number(await startField.inputValue());
  await endField.fill(String((start + 25).toFixed(2)));
  selection = await read();
  assert(Math.abs(selection.length - 20) < 0.06, `typed past the cap ${selection.length}`);
  // Digits typed into a field stay digits; leave the field and the keys pick presets again.
  await readout.click();
  await page.keyboard.press('2');
  selection = await read();
  assert(
    selection.length <= 15 && selection.length >= 14.6,
    `15 s after typing ${selection.length}`,
  );
  await page.screenshot({ path: join(shots, 'reference-trim-dark.png') });

  // 7. Confirming makes the cut the reference and the transcript uses that cut.
  const confirmed = selection;
  await dialog.getByRole('button', { name: 'Use this passage', exact: true }).click();
  await dialog.waitFor({ state: 'hidden' });
  const row = page.getByText(/4\.\d-\d+\.\ds\.wav|\d+\.\d-\d+\.\ds\.wav/);
  await row.waitFor({ timeout: 10000 });
  await page.getByText(`${confirmed.length.toFixed(1)}s`, { exact: true }).waitFor();
  await page.screenshot({ path: join(shots, 'reference-trim-accepted.png') });
  for (let attempt = 0; attempt < 40 && transcribeBodies.length === 0; attempt++)
    await new Promise((done) => setTimeout(done, 250));
  assert.equal(
    transcribeBodies.length,
    1,
    'one transcript request for the trimmed clip; API traffic:\n' + apiLog.join('\n'),
  );
  const body = transcribeBodies[0];
  assert(body.includes('RIFF'), 'transcript body is a WAV');
  assert(body.includes('4.wav') || body.includes('s.wav'), 'transcript body names the cut');
  const expectedWav = 44 + Math.round(confirmed.length * REFERENCE_RATE) * 2;
  const originalBytes = statSync(clip).size;
  assert(
    Math.abs(body.length - expectedWav) < 2500,
    `transcript body ${body.length} B, expected about ${expectedWav} B for ${confirmed.length} s`,
  );
  assert(Math.abs(body.length - originalBytes) > 2500, 'transcript did not get the original file');

  // 8. A clip within the limit still offers Trim on request.
  await page.getByRole('button', { name: 'Trim', exact: true }).click();
  await dialog.waitFor();
  await dialog.getByRole('button', { name: 'Use this passage', exact: true }).waitFor({
    timeout: 30000,
  });
  await page.keyboard.press('Escape');
  await dialog.waitFor({ state: 'hidden' });

  assert.deepEqual(errors, [], 'page errors');

  // 9. The same dialog in the light theme, for the visual check.
  const light = await browser.newContext({ viewport: { width: 1280, height: 860 } });
  const lightPage = await light.newPage();
  await lightPage.addInitScript(() => {
    localStorage.setItem('voicestudio.setup.complete.v1', '1');
    localStorage.setItem('voicestudio.locale', 'en');
    localStorage.setItem('voicestudio.theme', 'light');
  });
  await lightPage.route('**/api/engines', (route) => route.fulfill({ json: engines }));
  await lightPage.route('**/api/profiles', (route) => route.fulfill({ json: [] }));
  await lightPage.goto(origin + '/#/clone');
  await lightPage.getByText('Drop audio here', { exact: false }).waitFor({ timeout: 30000 });
  await lightPage.locator('input[type="file"]').first().setInputFiles(clip);
  const lightDialog = lightPage.getByRole('dialog');
  await lightDialog.getByRole('button', { name: 'Use this passage', exact: true }).waitFor({
    timeout: 30000,
  });
  await lightDialog.locator('[part~="region"]').waitFor();
  await lightPage.screenshot({ path: join(shots, 'reference-trim-light.png') });
  await light.close();

  console.log(
    `Reference trimmer smoke passed: auto-open on ${originalBytes} B clip, presets 10/15/20 + keys, ` +
      `drag, cap, snap, loop, play, ${confirmed.length} s cut confirmed and transcribed ` +
      `(${body.length} B WAV). Screenshots in ${shots}`,
  );
} finally {
  await browser.close();
  server.kill('SIGTERM');
  await backend.close();
}
