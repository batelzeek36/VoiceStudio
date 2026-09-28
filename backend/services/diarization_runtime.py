"""Persisted selection and installed-only resolution for diarisation runtimes."""
from __future__ import annotations

import os
from pathlib import Path

from core import prefs

PYANNOTE = "pyannote"
SORTFORMER = "audiocpp-sortformer"
SORTFORMER_REPO = "audio-cpp/audio.cpp-gguf"
SORTFORMER_FILE = "Sortformer-Diar-4spk-v1-GGUF/sortformer-diar-4spk-v1-q8_0.gguf"

_SORTFORMER_MODEL_MISSING = "Install the Sortformer model bundle"
_SORTFORMER_MODEL_BROKEN = "Repair the installed Sortformer model bundle"
_SORTFORMER_RUNTIME_MISSING = (
    "The Sortformer model is installed. Install the audio.cpp runtime to use it"
)
_SORTFORMER_CLI_MISSING = (
    "The installed audio.cpp runtime does not include speaker diarisation"
)


def selected_backend() -> str:
    value = str(
        prefs.resolve(
            "diarization_backend",
            env="OMNIVOICE_DIARIZATION_BACKEND",
            default=PYANNOTE,
        )
    ).strip()
    return value if value in {PYANNOTE, SORTFORMER} else PYANNOTE


def sortformer_model_path() -> Path:
    """The Sortformer GGUF as it is named: absolute, symlinks NOT resolved.

    The Hugging Face cache on macOS and Linux names the file in
    ``snapshots/<rev>/...q8_0.gguf`` as a symlink to ``blobs/<sha256>``, which
    has no extension. Resolving here handed every caller the blob, so the
    ``.gguf`` check below failed forever and a complete download still read
    "Install the Sortformer model bundle". Opening the named path follows the
    link; :func:`check_sortformer_model` resolves only to open it.
    """
    configured = os.environ.get("OMNIVOICE_DIARIZATION_MODEL", "").strip()
    if configured:
        return Path(os.path.abspath(Path(configured).expanduser()))

    # An installed-only lookup never reaches the network. Installation remains
    # an explicit Model Library action through the reviewed audio.cpp bundle.
    from huggingface_hub import hf_hub_download
    from services.hf_revisions import revision_for

    return Path(
        hf_hub_download(
            repo_id=SORTFORMER_REPO,
            filename=SORTFORMER_FILE,
            revision=revision_for(SORTFORMER_REPO),
            local_files_only=True,
        )
    )


def check_sortformer_model(model: Path) -> None:
    """Accept only an existing GGUF named ``*.gguf``.

    The name is judged on the path as named (see :func:`sortformer_model_path`:
    a resolved cache path is an extensionless blob); the file itself is
    opened through its resolved target and must start with the GGUF magic.
    Raises ``FileNotFoundError`` when it is absent or not named ``.gguf`` and
    ``ValueError`` when its contents are not GGUF; other ``OSError``s pass
    through. Messages carry no path.
    """
    if model.suffix.lower() != ".gguf":
        raise FileNotFoundError("The configured Sortformer model is not a .gguf file")
    target = model.resolve()
    if not target.is_file():
        raise FileNotFoundError("The configured Sortformer GGUF is missing")
    with target.open("rb") as model_file:
        if model_file.read(4) != b"GGUF":
            raise ValueError("The configured Sortformer model is not a GGUF file")


def select_backend(backend: str) -> None:
    if backend not in {PYANNOTE, SORTFORMER}:
        raise ValueError("Unknown diarisation engine")
    prefs.set_("diarization_backend", backend)


def sortformer_status() -> dict:
    """Return path-free readiness for the model and its native executable."""
    status = {
        "model": SORTFORMER_REPO,
        "model_installed": False,
        "runtime_installed": False,
        "installed": False,
        "reason": _SORTFORMER_MODEL_MISSING,
    }
    try:
        model = sortformer_model_path()
    except Exception:
        return status
    try:
        check_sortformer_model(model)
    except FileNotFoundError:
        return status
    except (OSError, ValueError):
        status["reason"] = _SORTFORMER_MODEL_BROKEN
        return status

    status["model_installed"] = True
    status["reason"] = _SORTFORMER_RUNTIME_MISSING
    try:
        from engines.audiocpp.bootstrap import resolve_server_binary

        server = resolve_server_binary()
    except (OSError, RuntimeError):
        return status
    cli = server.with_name("audiocpp_cli.exe" if os.name == "nt" else "audiocpp_cli")
    if not cli.is_file() or (os.name != "nt" and not os.access(cli, os.X_OK)):
        status["reason"] = _SORTFORMER_CLI_MISSING
        return status
    status.update(runtime_installed=True, installed=True, reason=None)
    return status


def installed_backends() -> set[str]:
    """Return complete local runtimes without loading weights or downloading."""
    installed: set[str] = set()
    from api.routers.setup.models import KNOWN_MODELS, cache_is_complete, is_cached

    repo_id = "pyannote/speaker-diarization-3.1"
    spec = next(model for model in KNOWN_MODELS if model["repo_id"] == repo_id)
    if is_cached(repo_id) and cache_is_complete(spec):
        installed.add(PYANNOTE)

    try:
        native = sortformer_status()
        if not native["installed"]:
            return installed
        installed.add(SORTFORMER)
    except (OSError, RuntimeError, ValueError):
        pass
    return installed
