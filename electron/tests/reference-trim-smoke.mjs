/**
 * Reference trimmer smoke: a real over-long clip dropped on Voice Clone opens
 * the trimmer; presets, keys, drag, on-waveform playback, snap, loop, zoom
 * containment, the 20 s cap, the confirm and the reopen-on-original are
 * driven for real at 1440 wide, and the transcript request is checked to
 * carry the trimmed WAV rather than the original file.
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
const VIEWPORT = { width: 1440, height: 900 };

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
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
const lengthPattern = /^(\d+\.\d) s$/;

const browser = await chromium.launch({
  channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
  headless: true,
});

async function openClone(context) {
  const page = await context.newPage();
  await page.addInitScript(() => {
    localStorage.setItem('voicestudio.setup.complete.v1', '1');
    localStorage.setItem('voicestudio.locale', 'en');
  });
  await page.route('**/api/engines', (route) => route.fulfill({ json: engines }));
  await page.route('**/api/profiles', (route) => route.fulfill({ json: [] }));
  await page.goto(origin + '/#/clone');
  await page.getByText('Drop audio here', { exact: false }).waitFor({ timeout: 30000 });
  return page;
}

async function openTrimmer(page) {
  const dialog = page.getByRole('dialog');
  await dialog.waitFor({ timeout: 20000 });
  await dialog.getByRole('button', { name: 'Use this passage', exact: true }).waitFor({
    timeout: 30000,
  });
  await dialog.locator('[part~="region"]').waitFor();
  await sleep(300);
  return dialog;
}

try {
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await openClone(context);
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const transcribeBodies = [];
  await page.route('**/api/transcribe', (route) => {
    transcribeBodies.push(route.request().postDataBuffer());
    return route.fulfill({ json: { text: 'and yes dear I am using myself as the example' } });
  });

  // 1. An over-long pick opens the trimmer by itself on a window from the first speech.
  await page.locator('input[type="file"]').first().setInputFiles(clip);
  const dialog = await openTrimmer(page);
  const readout = dialog.getByText(lengthPattern);
  const read = async () => {
    const match = (await readout.innerText()).match(lengthPattern);
    assert(match, 'length readout missing');
    return Number(match[1]);
  };
  const inView = async () => {
    const box = await dialog.boundingBox();
    assert(box, 'dialog box');
    const confirm = await dialog
      .getByRole('button', { name: 'Use this passage', exact: true })
      .boundingBox();
    assert(confirm, 'confirm box');
    assert(
      box.x >= 0 &&
        box.x + box.width <= VIEWPORT.width &&
        confirm.x + confirm.width <= box.x + box.width + 1,
      `dialog ran off screen: dialog right ${box.x + box.width}, confirm right ${confirm.x + confirm.width}`,
    );
    return box.width;
  };
  let length = await read();
  assert(length <= 15 && length >= 14.6, `default window ${length}`);
  const dialogWidth = await inView();
  // No exact seconds, switches, zoom or key legend on the calm surface.
  assert.equal(await dialog.getByRole('spinbutton').count(), 0, 'fields hidden until Fine-tune');
  assert.equal(await dialog.getByRole('switch').count(), 0, 'switches hidden until Fine-tune');
  await page.screenshot({ path: join(shots, 'reference-trim-open.png') });

  // 2. Presets keep the start and set the length; 1/2/3 do the same.
  const startField = dialog.getByRole('spinbutton', { name: 'Start', exact: true });
  const endField = dialog.getByRole('spinbutton', { name: 'End', exact: true });
  await dialog.getByRole('button', { name: /^10 s$/ }).click();
  length = await read();
  assert(length <= 10 && length >= 9.6, `10 s preset ${length}`);
  await page.keyboard.press('3');
  length = await read();
  assert(length <= 20 && length >= 19.6, `20 s key ${length}`);
  await page.keyboard.press('2');
  length = await read();
  assert(length <= 15 && length >= 14.6, `15 s key ${length}`);
  await page.keyboard.press('1');
  length = await read();
  assert(length <= 10 && length >= 9.6, `10 s key ${length}`);

  // 3. Play runs the selection on the waveform itself: the playhead moves inside the region.
  const wave = dialog.locator('[data-slot="audio-trimmer-wave"]');
  const region = dialog.locator('[part~="region"]');
  const cursor = dialog.locator('[part~="cursor"]');
  const play = dialog.getByRole('button', { name: 'Play', exact: true });
  await play.click();
  await dialog.getByRole('button', { name: 'Pause', exact: true }).waitFor({ timeout: 10000 });
  await sleep(1200);
  const regionBox = await region.boundingBox();
  const cursorBox = await cursor.boundingBox();
  assert(regionBox && cursorBox, 'region and cursor visible while playing');
  assert(
    cursorBox.x > regionBox.x + 4 && cursorBox.x <= regionBox.x + regionBox.width + 2,
    `playhead ${cursorBox.x} outside the selection ${regionBox.x} to ${regionBox.x + regionBox.width}`,
  );
  await page.screenshot({ path: join(shots, 'reference-trim-playing.png') });
  // Space pauses even though the Play button holds focus.
  await page.keyboard.press(' ');
  await play.waitFor({ timeout: 5000 });

  // 4. Dragging on empty waveform draws a new selection; a long drag is capped.
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
  length = await read();
  assert(length >= 5 && length <= 8.5, `drag length ${length}`);
  await drag(0.05, 0.95);
  length = await read();
  assert(length <= 20, `capped drag ${length}`);
  await page.keyboard.press('1');

  // 5. Fine-tune: exact seconds, snap, loop, zoom, and the keys legend.
  await dialog.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await startField.waitFor();
  const snap = dialog.getByRole('switch', { name: 'Snap to silence' });
  const loop = dialog.getByRole('switch', { name: 'Loop preview' });
  assert.equal(await snap.getAttribute('aria-checked'), 'true');
  assert.equal(await loop.getAttribute('aria-checked'), 'true');
  await snap.click();
  assert.equal(await snap.getAttribute('aria-checked'), 'false');
  await page.keyboard.press('2');
  assert.equal(await read(), 15, 'unsnapped preset is exact');
  await snap.click();
  await page.keyboard.press('2');
  length = await read();
  assert(length <= 15, `snapped preset ${length}`);
  await loop.click();
  assert.equal(await loop.getAttribute('aria-checked'), 'false');
  await loop.click();
  // Typed seconds commit on Enter, and the cap holds.
  const start = Number(await startField.inputValue());
  await endField.fill(String((start + 25).toFixed(2)));
  await endField.press('Enter');
  length = await read();
  assert(Math.abs(length - 20) < 0.06, `typed past the cap ${length}`);
  // A digit typed into a field is a digit; afterwards the keys pick presets again.
  await startField.fill('7');
  assert.equal(await read(), 20, 'typing did not move the selection');
  await startField.press('Enter');
  await page.keyboard.press('2');
  length = await read();
  assert(length <= 15 && length >= 14.6, `15 s after typing ${length}`);
  await page.screenshot({ path: join(shots, 'reference-trim-fine-tune.png') });

  // 6. Zoom stays inside the dialog at 1440.
  await dialog.getByRole('button', { name: 'Zoom in (+)', exact: true }).click();
  await dialog.getByRole('button', { name: 'Zoom in (+)', exact: true }).click();
  await sleep(300);
  assert.equal(await inView(), dialogWidth, 'dialog width unchanged after zoom');
  await page.screenshot({ path: join(shots, 'reference-trim-zoomed.png') });
  await page.keyboard.press('End');
  await sleep(200);
  await inView();
  await page.keyboard.press('Home');
  await sleep(200);
  await inView();
  const confirmedStart = Number(await startField.inputValue());
  await dialog.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await page.screenshot({ path: join(shots, 'reference-trim-dark.png') });

  // 7. Confirming makes the cut the reference and the transcript uses that cut.
  const confirmedLength = await read();
  await dialog.getByRole('button', { name: 'Use this passage', exact: true }).click();
  await dialog.waitFor({ state: 'hidden' });
  await page.getByText(/\d+\.\d-\d+\.\ds\.wav/).waitFor({ timeout: 10000 });
  await page.getByText(`${confirmedLength.toFixed(1)}s`, { exact: true }).waitFor();
  await page.screenshot({ path: join(shots, 'reference-trim-accepted.png') });
  for (let attempt = 0; attempt < 40 && transcribeBodies.length === 0; attempt++) await sleep(250);
  assert.equal(transcribeBodies.length, 1, 'one transcript request for the trimmed clip');
  const body = transcribeBodies[0];
  assert(body.includes('RIFF'), 'transcript body is a WAV');
  const expectedWav = 44 + Math.round(confirmedLength * REFERENCE_RATE) * 2;
  const originalBytes = statSync(clip).size;
  assert(
    Math.abs(body.length - expectedWav) < 2500,
    `transcript body ${body.length} B, expected about ${expectedWav} B for ${confirmedLength} s`,
  );
  assert(Math.abs(body.length - originalBytes) > 2500, 'transcript did not get the original file');

  // 8. Trim reopens the WHOLE original with the last cut preselected.
  await page.getByRole('button', { name: 'Trim', exact: true }).click();
  const reopened = await openTrimmer(page);
  assert.equal(await read(), confirmedLength, 'reopened on the last cut');
  await reopened.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await startField.waitFor();
  const total = await reopened.getByText(/^of \d+\.\d\d s$/).innerText();
  assert(Number(total.match(/of (\d+\.\d\d) s/)[1]) > 30, `whole original is back: ${total}`);
  assert(
    Math.abs(Number(await startField.inputValue()) - confirmedStart) < 0.02,
    'last cut preselected',
  );
  await page.screenshot({ path: join(shots, 'reference-trim-reopened.png') });
  await page.keyboard.press('Escape');
  await reopened.waitFor({ state: 'hidden' });
  await page.getByText(`${confirmedLength.toFixed(1)}s`, { exact: true }).waitFor();

  // 9. Cancelling a demanded trim says so.
  await page.getByRole('button', { name: 'Clear', exact: true }).click();
  await page.getByText('Drop audio here', { exact: false }).waitFor();
  await page.locator('input[type="file"]').first().setInputFiles(clip);
  await openTrimmer(page);
  await page.keyboard.press('Escape');
  await page.getByText('The clip was not kept', { exact: false }).waitFor({ timeout: 5000 });
  await page.getByText('Drop audio here', { exact: false }).waitFor();

  assert.deepEqual(errors, [], 'page errors');

  // 10. The same dialog in the light theme, for the visual check.
  const light = await browser.newContext({ viewport: VIEWPORT });
  const lightPage = await light.newPage();
  await lightPage.addInitScript(() => localStorage.setItem('voicestudio.theme', 'light'));
  await lightPage.addInitScript(() => {
    localStorage.setItem('voicestudio.setup.complete.v1', '1');
    localStorage.setItem('voicestudio.locale', 'en');
  });
  await lightPage.route('**/api/engines', (route) => route.fulfill({ json: engines }));
  await lightPage.route('**/api/profiles', (route) => route.fulfill({ json: [] }));
  await lightPage.goto(origin + '/#/clone');
  await lightPage.getByText('Drop audio here', { exact: false }).waitFor({ timeout: 30000 });
  await lightPage.locator('input[type="file"]').first().setInputFiles(clip);
  const lightDialog = await openTrimmer(lightPage);
  await lightDialog.getByRole('button', { name: 'Play', exact: true }).click();
  await lightDialog.getByRole('button', { name: 'Pause', exact: true }).waitFor({ timeout: 10000 });
  await sleep(900);
  await lightPage.screenshot({ path: join(shots, 'reference-trim-light.png') });
  await light.close();

  console.log(
    `Reference trimmer smoke passed at ${VIEWPORT.width} wide: auto-open on ${originalBytes} B clip, ` +
      `presets 10/15/20 + keys, on-waveform playback, drag, cap, snap, loop, zoom contained, ` +
      `${confirmedLength} s cut confirmed and transcribed (${body.length} B WAV), reopened on the ` +
      `original, cancelled demand toasted. Screenshots in ${shots}`,
  );
} finally {
  await browser.close();
  server.kill('SIGTERM');
  await backend.close();
}
