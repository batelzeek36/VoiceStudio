# Generation Parameters

Parameters can be passed as keyword arguments to `model.generate(...)` or via the `OmniVoiceGenerationConfig` dataclass. See below for the full list and which category each belongs to.

```python
# 1) Direct keyword arguments
audio = model.generate(text="Hello world", num_step=32, guidance_scale=2.0)

# 2) Via OmniVoiceGenerationConfig dataclass
from omnivoice import OmniVoiceGenerationConfig

config = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0)
audio = model.generate(text="Hello world", generation_config=config)
```

## Decoding

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_step` | int | 32 | Number of iterative unmasking steps. Higher values improve quality but slow down generation. Use 16 for faster inference. |
| `denoise` | bool | True | Prepend the `<|denoise|>` token to the input, which signals the model to produce cleaner speech. |
| `guidance_scale` | float | 2.0 | Classifier-free guidance scale.|
| `t_shift` | float | 0.1 | Time-step shift for the noise schedule. Smaller values emphasise earlier steps in decoding. |

## Sampling

| Parameter | Type | Default | Description |
|---|---|---|---|
| `position_temperature` | float | 5.0 | Temperature for mask-position selection. 0 = greedy (deterministic). Higher values increase randomness. |
| `class_temperature` | float | 0.0 | Temperature for token sampling at each step. 0 = greedy (deterministic). Higher values increase randomness. |
| `layer_penalty_factor` | float | 5.0 | Penalty applied to deeper codebook layers, encouraging earlier (lower) layers to unmask first. |

> Using temperature (and seed pinning) to elicit expressive delivery — breaths, laughter, sighs — is covered in [expressive-speech.md](expressive-speech.md), including the tradeoffs.

## Duration & Speed

These accept a single value applied to all items, or a per-item list (useful in batch mode):

```python
# Fixed 10-second output
audio = model.generate(text="Hello, this is a test of duration control", duration=10.0)

# Faster speech (1.2x faster than estimated)
audio = model.generate(text="Hello, this is a test of duration control", speed=1.2)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `duration` | float or list[float \| None] | None | Fixed output duration in seconds. Overrides `speed` when set. |
| `speed` | float or list[float \| None] | None | Speed factor. Values > 1.0 produce shorter audio (faster); values < 1.0 produce longer audio (slower). Ignored when `duration` is set. Defaults to 1.0 when both are None. |

Priority: `duration` > `speed`.

## Pre/Post Processing

| Parameter | Type | Default | Description |
|---|---|---|---|
| `preprocess_prompt` | bool | True | Whether to apply preprocessing to the voice-clone prompt audio (remove long silences in reference audio, add punctuation in the end of reference text). |
| `postprocess_output` | bool | True | Apply post-processing to generated audio (remove long silences). |

> **Note — quiet recordings are safe.** Silence removal is adaptive: if trimming at the standard threshold would consume a quiet-but-real recording, progressively gentler thresholds are tried and, as a last resort, trimming is skipped — a quiet clip clones instead of erroring. Only a clip with genuinely no audio (empty or fully silent) is rejected, with guidance to re-record closer to the microphone.
>
> **Tip — reference-clip quality transfers.** Zero-shot cloning mirrors the acoustics of the reference clip, not just the voice: a clip recorded in an echoey room clones echoey. Record dry and close-mic for clean output. No effect preset adds reverb unless you choose one that declares it (Cinematic, Warm).

For an in-app recording, choose the microphone and Auto, Mono, or Stereo in the Voice panel. While recording, the input meter confirms whether VoiceStudio is receiving a usable signal; monitoring is visual and never plays the microphone through the speakers.

## Long-Form Generation

To support stable long-form speech generation with low VRAM consumption, the text is automatically split into smaller segments when the estimated duration of the generated speech exceeds `audio_chunk_duration`, with each segment producing approximately `audio_chunk_duration` seconds of audio. This approach allows the model to accept arbitrarily long text and generate arbitrarily long speech with near-constant VRAM consumption.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_chunk_duration` | float | 15.0 | Target chunk duration (seconds) when splitting long text. |
| `audio_chunk_threshold` | float | 30.0 | Estimated audio duration (seconds) above which chunking is activated. |

## How long will it take?

VoiceStudio estimates render time from renders on your own machine, never from a built-in figure: the same chapter can take minutes on a CUDA desktop and an hour on a CPU-only laptop, and `num_step` alone changes it several times over.

- **What is measured.** Every finished synthesis call records the engine, the device it ran on (`cuda`, `rocm`, `mps`, `directml`, `cpu`, ...), the steps, the characters, the audio it produced, its speed, the length of the reference audio it was conditioned on, whether it was a cold call, and how long it took. Numbers only: no text, voice or file path. The newest 200 calls per engine, device and steps are kept in the local database and never sent anywhere, analytics and bug reports included.
- **How the estimate is made.** It plans the calls the render will make with the render's own code: every `[voice:]` switch, `[pause]`, inline markup boundary and chunk is a separate call with its own fixed cost. Each call's audio length comes from OmniVoice's own duration estimate for the chosen voice, or from this machine's measured characters per second for other engines. Each call is then priced from this machine's calls: OmniVoice renders a call over 30 s of audio as passes of about 15 s, and each pass carries the voice's reference audio, so the model fits one robust line per pass, `wall = a + b * (pass audio + reference)` (Theil-Sen slope, median intercept); other engines use the call's audio. The range is how far this machine's past calls strayed from what the line predicted for them, never tighter than 10 percent.
- **Your machine right now.** Throughput drifts with what else the computer is doing, so the newest calls set the current pace (each older call counts half as much per two calls, and one odd call cannot move it). After an idle stretch the last pace is kept but the range widens to cover this machine's usual speed too; a machine that is behaving erratically gets a wider range.
- **Cold starts.** The first call after the engine loads (at startup, or after an idle engine was unloaded) and the first call on a reference longer than 20 s (while its best 15 s passage is chosen) cost extra. Those calls are kept out of the line and measured on their own, and the estimate adds that cost only when the render will pay it: the engine is not loaded right now, or a voice in the script has not been used yet.
- **Measured, rough or not yet.** *Measured*: at least three calls on this engine and device at these steps. *Rough*: measured at other steps and rescaled by the steps ratio (compute grows with unmasking steps; the per-call cost does not), a planned call longer or shorter than anything measured, or a cold start not measured yet; the range is wider. *Not yet*: fewer than three calls on this engine and device, so the first few calls calibrate it (one long take or a chapter can be enough; three short takes always are) and the app says so instead of guessing.
- **Where it appears.** Beside Synthesize in Voice cloning, after the runtime estimate in Stories and Audiobook, and as a countdown while a render runs. Stories and Audiobook re-fit the countdown after every engine call (the render streams `progress` frames with the seconds left), so even a one-chapter book is corrected as it goes; Voice cloning counts its take down from the estimate. A render sent to a remote worker gets no estimate from this machine. Model loading is not included.
- **API.** `POST /render/estimate` takes what the render takes (`surface`: `generate` with the `/generate` fields, `audiobook` with the `/audiobook` body, `longform` with the `/longform/render` body) and returns `seconds`, `low`, `high`, `calls`, `samples`, `basis` (`measured`, `rough` or `none`), `reason` (`cold_start`, `remote` or `no_rate` when there is no number) and per-chapter `parts`. It loads no model and makes no network call.
