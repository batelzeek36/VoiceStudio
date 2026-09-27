# Electron privacy and retention

Settings > Privacy exposes the existing invisible-watermark setting, analytics consent and generation-history retention. Render timings behind the render-time estimate (engine, device, steps, characters, seconds; never text, voices or paths) stay in the local database and are excluded from analytics and bug reports.

Watermark controls appear when the backend reports AudioSeal available and affect new audio through the existing marking path. No audio producer bypasses `mark_synthetic`. Analytics consent uses the existing backend opt-in endpoint and remains unchanged until the user explicitly switches it. The Electron renderer does not initialize an additional analytics SDK; the setting controls backend analytics. Unavailable features do not display inert toggles, and failed saves preserve confirmed state.

Translation privacy classification is shared with Tauri. Unknown or unavailable backend data does not claim offline operation. The existing `libretranslate` identifier is an alias for the backend's local Argos branch, not an assumption about an external LibreTranslate service. The Translation link opens engine settings.

Retention supports 0 (unlimited), validates whole-number limits up to 100000 and explains that cleanup removes old unstarred takes and their audio after generation. A more restrictive cap requires inline confirmation. Raising the cap or disabling cleanup saves directly. No records or files are deleted by simply opening settings or editing the input.

`electron/tests/privacy-settings-smoke.mjs` covers consent, watermark changes, failed writes, availability, retention confirmation/cancel, unlimited retention and reload with mocked endpoints. Tauri privacy regressions pass after sharing classification. Live read-only checks verified the watermark, analytics and retention schemas without changing the user's privacy preferences or deleting data.

AudioSeal receives 16 kHz audio for embedding and detection, including with
AudioSeal 0.2, which no longer resamples internally. Processing remains bounded
by chunk size. Only the watermark residual is resampled back and added to the
original signal, retaining its sample rate, length, and high-frequency content.
