"""The text a /generate render hands its engine, prepared in one place.

``POST /generate`` and the render-time estimate (``POST /render/estimate``)
both call :func:`prepare_synthesis_text`, so the estimate plans exactly the
text the render will synthesize: same normalization, same pronunciation
dictionary, same chunk boundaries downstream.
"""
from __future__ import annotations

import os
from typing import Optional


def pronunciation_enabled() -> bool:
    """The user pronunciation dictionary's switch: ``OMNIVOICE_PRONUNCIATION``
    (a power-user override; "0"/"false"/"no"/"off" disable it) wins over the
    ``pronunciation_enabled`` pref, which defaults ON."""
    env = os.environ.get("OMNIVOICE_PRONUNCIATION")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "off", "")
    from core import prefs

    return bool(prefs.get("pronunciation_enabled", True))


def prepare_synthesis_text(text: str, language: Optional[str], *, pronounce: bool = True) -> str:
    """Normalize, then apply the pronunciation dictionary.

    Engine-agnostic text normalization (junk strip, numbers to words,
    abbreviations) runs AFTER ``language`` is fully resolved and BEFORE the
    pronunciation dictionary, so user dictionary entries operate on normalized
    text and respellings are never re-mangled (ordering rationale in
    services/text_normalization.py). Pref-gated (default ON), idempotent,
    never raises.

    Expressive-TTS Spec 01: the user pronunciation dictionary and inline
    ``[[...]]`` one-off overrides apply here, after ``language`` is resolved (a
    profile may fill it) so per-language entries match the real render
    language, and before the text reaches either inference path and the chunk
    splitter. Pure text substitution, identical on macOS, Windows and Linux. A
    disabled pref or empty dictionary is a pass-through, so plain text stays
    byte-identical (#G5 backward-compat).
    """
    from services.text_normalization import normalize_for_tts

    text = normalize_for_tts(text, language)
    if pronounce and pronunciation_enabled():
        from services.pronunciation import apply_pronunciation, load_entries_from_db

        try:
            rows = load_entries_from_db()
        except Exception:  # noqa: BLE001 - table missing / DB locked: no-op
            rows = []
        return apply_pronunciation(text, rows, language)
    # Even with the dictionary off, inline [[...]] overrides are an explicit,
    # in-text authoring choice: always honored (and never left as literal
    # double-bracket text the model would mispronounce).
    from services.pronunciation import apply_inline_overrides

    return apply_inline_overrides(text)
