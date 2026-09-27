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
