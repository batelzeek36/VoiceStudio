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
   `audio_seconds` produced, `wall_seconds`, `speed`, `ref_seconds` (the reference audio the call was conditioned on),
   `cold` (see 2), `created_at`. Numbers and identifiers only: no text, no voice, no paths
   (the same rule as `render_trace`). Kept to the most recent 200 rows per `(engine, device, num_step)`. Never sent
   anywhere: excluded from analytics and bug reports. Recorded at the one place each surface already calls
   `trace_call("synthesis", ...)` and knows the sample rate, through one helper (`services/render_timing.py`).

2. **Model.** A call's cost follows the sequence the model attends over, not its audio length alone. Measured on an
   M1 Max (OmniVoice on MPS, 64 steps), a single line on audio seconds missed held-out calls by a median 18 to 26
   percent, because OmniVoice generates a call whose estimated audio exceeds 30 s as passes of about 15 s
   (`audio_chunk_threshold`, `audio_chunk_duration`), each paying the fixed cost again, and every pass carries the
   voice's reference audio in its sequence (a 15 s reference costs more per call than an 11 s one). So each call is
   split into passes and one robust line is fitted per pass, per bucket `(engine, device, num_step)`:
   `wall_per_pass = a + b * length`, where `length` is the pass's audio plus, for OmniVoice, its reference (other
   engines: the call's audio, one pass). Theil-Sen slope, median intercept, `a >= 0`, `b > 0`, on the newest warm
   calls. The range is the 10th to 90th percentile of cross-validated errors (each call predicted by a line fitted
   without it), never tighter than 10 percent either way. Missing bucket, same `(engine, device)` with other steps:
   scale `b` by the steps ratio (compute is proportional to unmasking steps; the per-pass cost `a` is not), and mark
   the result *rough*. A planned pass more than 10 percent shorter or longer than anything measured is also *rough*,
   with a wider range. Fewer than 3 warm calls on this machine for that engine and device: **no number** ("Estimate
   appears after a few renders on this machine"), because any prior would be wrong somewhere.

   **Cold calls** are flagged when recorded, from the engine's real state, and kept out of the line: `load` is the
   first call on a freshly loaded engine (a new sidecar process, a new native model object, a reloaded adapter model;
   a sidecar reaped for idleness comes back cold); `voice` is the first call on a reference longer than the engine's
   `max_ref_seconds` whose 15 s passage has not been chosen yet. Each kind's overhead is measured as what its cold
   calls took beyond the warm line, and the estimate adds it only when the planned render will pay it (the engine is
   not loaded right now; a voice in the plan is unranked). A cold cost whose signal is read but never measured here
   widens the range and marks the estimate rough instead. Not cold, by reading the code and the data: the OmniVoice
   sidecar has no prompt cache and encodes the reference on every call, so a longer reference is cost on every call
   (modelled as length above), and the native prompt cache saves about 0.4 s per new voice, below timing noise.

   **Recency.** A desktop's throughput drifts with its background load: on the M1 Max the same call took 35.7 s and,
   minutes later under heavier load from other work, 51.0 s. The line keeps the shape; the machine's current *pace*
   multiplies it: the weighted mean of the recent calls' log(observed / predicted), each weighted by
   `2 ** -(age / 2 min)` of wall-clock age, with a prior of half a call at pace 1. Wall-clock, not call count, because
   what drifts is load, which moves with time: after an idle hour the old evidence has aged out and the estimate
   returns to the long-run line, while during a burst of calls the newest three to six decide. Recent calls that
   scatter widen the range (their weighted 10th to 90th percentile around the paced line).

3. **Plan the calls.** `POST /render/estimate` takes what a render takes (surface, text or script, voice map, engine,
   steps, speed, duration, chunking) and runs the same planning code the render runs (pause markers, the longform
   parser, markup splits, the chunker), so the number of calls and the audio each produces match the real render.
   Audio seconds per call: the engine's own duration estimate where it has one (OmniVoice), otherwise characters per
   second learned from this machine's `render_timings` rows for that engine. Response: `seconds`, `low`, `high`,
   `calls`, `samples`, `basis` (`measured` | `rough` | `none`), `warmup_seconds` (the cold costs included). Pure
   local compute, no model load.

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
  `none` basis, call counting for markup and chunking, migration upgrade, frontend formatting, and eight real calls
  from one machine as a numbers-only fixture: leave-one-out over its seven warm calls must miss by at most a median
  20 percent with the actual inside the range for at least five (measured: 3.4 percent, six of seven; the line this
  replaced: 17.8 percent, three of seven).
- Docs: `docs/` generation-parameters page gets a short "How long will it take?" section; CHANGELOG `[Unreleased]`
  one-liner.
- Old installs start with an empty table and show "after your first render" once, then estimates.

## Implementation notes

Refinements made while building it, all within the decision above:

- `render_timings` stores `speed` (a number), so characters per second can be learned at speed 1. Besides the 200
  rows per bucket there is a 5,000-row backstop across all buckets.
- Live check on the same M1 Max after the change (profile with an 11.35 s reference, 64 steps): a call on a sidecar
  reaped for idleness was estimated at 27.9 s including 7.0 s of measured load and took 29.0 s; the next call, a new
  27 s read, was estimated at 36.5 s (range 32.8 to 47.8) and took 36.7 s.
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
- Code: `services/render_timing.py` (record), `services/render_warmth.py` (cold-call signals),
  `services/render_fit.py` (model and recency), `services/render_plan.py` and
  `services/render_estimate.py` with `api/routers/render_estimate.py` (plan and price), and the Electron renderer's
  `render-estimate-hint.tsx`, `render-estimate-inline.tsx` and `generation-progress.tsx` (show).
