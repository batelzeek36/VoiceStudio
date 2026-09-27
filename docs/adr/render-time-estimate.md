# Render-time estimate: measured on this machine, shown before and during a render

**Status:** Proposed
**Date:** 2026-09-27

## Context

The Audiobook screen shows the *audio* length a script will make (`AUDIOBOOK_WPM = 155`), but nothing tells a user
how long the *render* will take, and on the Voice cloning screen there is no hint at all. Render time varies by two
orders of magnitude between machines (a CUDA desktop, an Apple Silicon laptop on MPS, a CPU-only box) and, on one
machine, by settings: OmniVoice's cost scales with unmasking steps, and every `[voice:]` switch, `[pause]`, markup
boundary and ~15 s chunk is a separate model call with its own fixed cost. A number baked into the app (for example
"4x real time") would be right on one machine and wrong on every other one.

What exists: `core/render_trace.py` already times every `synthesis` call of a render, locally, numbers only, but keeps
only the last 32 renders in memory and does not record how much audio each call produced or under which settings.
`generation_history` stores `duration_seconds` and `generation_time` per Voice-cloning take, without engine, device or
steps. Measured on one M1 Max (MPS, OmniVoice, 64 steps): a 66 s read took 276 s.

## Decision

**Measure on the user's machine, never assume.**

1. **Record.** Every completed synthesis call on every surface (generate, audiobook, longform, preview) adds one row to
   a new local table `render_timings` (alembic migration): `engine`, `device` (the resolved torch device class:
   `cuda`, `mps`, `rocm`, `directml`, `cpu`), `num_step` (nullable for engines without steps), `text_chars`,
   `audio_seconds` produced, `wall_seconds`, `created_at`. Numbers and identifiers only: no text, no voice, no paths
   (the same rule as `render_trace`). Kept to the most recent 200 rows per `(engine, device, num_step)`. Never sent
   anywhere: excluded from analytics and bug reports. Recorded at the one place each surface already calls
   `trace_call("synthesis", ...)` and knows the sample rate, through one helper (`services/render_timing.py`).

2. **Model.** Per bucket `(engine, device, num_step)`: `wall = a + b * audio_seconds`, fitted robustly on the most
   recent samples (Theil-Sen slope, median intercept; clamp `a >= 0`, `b > 0`). The spread of residuals (25th to 75th
   percentile) gives the range. Missing bucket, same `(engine, device)` with other steps: scale `b` by the steps ratio
   (compute is proportional to unmasking steps; the per-call cost `a` is not), and mark the result *rough*. Fewer than
   3 samples on this machine for that engine and device: **no number** ("Estimate appears after your first render on
   this machine"), because any prior would be wrong somewhere.

3. **Plan the calls.** `POST /render/estimate` takes what a render takes (surface, text or script, voice map, engine,
   steps, speed, duration, chunking) and runs the same planning code the render runs (pause markers, the longform
   parser, markup splits, the chunker), so the number of calls and the audio each produces match the real render.
   Audio seconds per call: the engine's own duration estimate where it has one (OmniVoice), otherwise characters per
   second learned from this machine's `render_timings` rows for that engine. Response: `seconds`, `low`, `high`,
   `calls`, `samples`, `basis` (`measured` | `rough` | `none`). Pure local compute, no model load.

4. **Show it.** Voice cloning: beside Synthesize, "About 4 min to render" (debounced as the script or settings change).
   Stories and Audiobook: after the runtime estimate, "render about 13 min". During a render: a countdown from the
   planned total and the calls done so far, re-fitted as calls finish, never below zero ("finishing up" instead).
   Every string through i18n, in all locales.

## Why this works on other machines

- Nothing is hard-coded per device: the first render on a machine is the calibration, and the estimate says so until
  it has data. A fast GPU and a slow CPU each learn their own line.
- Buckets separate engines, devices and steps, so switching engine or GPU never mixes numbers.
- The call count comes from the real planning code, so markup-heavy scripts (many short calls) are costed right.
- Local-first: one local table, no network, no new dependency; behaviour is identical on macOS, Windows and Linux
  (only the numbers differ, by design).

## Consequences

- One alembic migration (new table, no change to existing ones); the upgrade path gets a test.
- Tests: the fit on synthetic samples (known `a`, `b`, noise, outliers), the steps-ratio fallback, the cold-start
  `none` basis, call counting for markup and chunking, migration upgrade, frontend formatting.
- Docs: `docs/` generation-parameters page gets a short "How long will it take?" section; CHANGELOG `[Unreleased]`
  one-liner.
- Old installs start with an empty table and show "after your first render" once, then estimates.

## Implementation notes

Refinements made while building it, all within the decision above:

- `render_timings` also stores `speed` (a number), so characters per second can be learned at speed 1. Besides the
  200 rows per bucket there is a 5,000-row backstop across all buckets.
- `num_step` is bucketed only for engines that declare `honors_num_step` (OmniVoice in process and in its sidecars,
  VoxCPM2, dots.tts, Supertonic-3); for every other engine it is NULL, so a value the engine ignores never splits its
  measurements. Adapters that host several models behind one id (mlx-audio, CosyVoice, sherpa-onnx, audio.cpp) are
  keyed by engine and model.
- Device is the routing answer for that engine on this host (`services/engine_routing`), plus `directml` for the native
  loader, computed the same way by the recorder and the estimator so a render and its estimate always name one
  bucket.
- The rough fit pools every sample of the engine and device, rescaling each one's audio by its steps ratio, which is
  the same as scaling `b` and leaving `a`; its range is widened by 20 to 25 percent.
- The response also carries `reason` (`cold_start`, `remote`, `no_rate`), `audio_seconds`, the bucket, and `parts`:
  planned seconds per chapter, which the live countdown re-fits as chapters finish (cached chapters count as instant,
  one slow chapter bends the rest at most 4x, the chapter in progress waits at zero instead of going negative).
- The calibration message reads "Estimate appears after a few renders on this machine": the threshold is three
  calls, and one short take is one call, so "after your first render" would be false after a first short take.
- Code: `services/render_timing.py` (record), `services/render_fit.py` (model), `services/render_plan.py` and
  `services/render_estimate.py` with `api/routers/render_estimate.py` (plan and price), and the Electron renderer's
  `render-estimate-hint.tsx`, `render-estimate-inline.tsx` and `generation-progress.tsx` (show).
