import os

from services import diarization_runtime


def _gguf(path):
    path.write_bytes(b"GGUF" + b"\0" * 16)
    return path


def test_sortformer_status_distinguishes_installed_model_from_missing_runtime(
    monkeypatch, tmp_path
):
    model = _gguf(tmp_path / "sortformer.gguf")
    monkeypatch.setattr(diarization_runtime, "sortformer_model_path", lambda: model)

    from engines.audiocpp import bootstrap

    def missing_runtime():
        raise RuntimeError("private path and setup diagnostic")

    monkeypatch.setattr(bootstrap, "resolve_server_binary", missing_runtime)

    status = diarization_runtime.sortformer_status()

    assert status == {
        "model": diarization_runtime.SORTFORMER_REPO,
        "model_installed": True,
        "runtime_installed": False,
        "installed": False,
        "reason": "The Sortformer model is installed. Install the audio.cpp runtime to use it",
    }


def test_sortformer_status_requires_matching_cli(monkeypatch, tmp_path):
    model = _gguf(tmp_path / "sortformer.gguf")
    server = tmp_path / ("audiocpp_server.exe" if os.name == "nt" else "audiocpp_server")
    server.write_bytes(b"server")
    monkeypatch.setattr(diarization_runtime, "sortformer_model_path", lambda: model)

    from engines.audiocpp import bootstrap

    monkeypatch.setattr(bootstrap, "resolve_server_binary", lambda: server)

    status = diarization_runtime.sortformer_status()

    assert status["model_installed"] is True
    assert status["runtime_installed"] is False
    assert status["installed"] is False
    assert status["reason"] == (
        "The installed audio.cpp runtime does not include speaker diarisation"
    )


def test_sortformer_status_reports_complete_runtime(monkeypatch, tmp_path):
    model = _gguf(tmp_path / "sortformer.gguf")
    server = tmp_path / ("audiocpp_server.exe" if os.name == "nt" else "audiocpp_server")
    cli = tmp_path / ("audiocpp_cli.exe" if os.name == "nt" else "audiocpp_cli")
    server.write_bytes(b"server")
    cli.write_bytes(b"cli")
    cli.chmod(0o755)
    monkeypatch.setattr(diarization_runtime, "sortformer_model_path", lambda: model)

    from engines.audiocpp import bootstrap

    monkeypatch.setattr(bootstrap, "resolve_server_binary", lambda: server)

    assert diarization_runtime.sortformer_status() == {
        "model": diarization_runtime.SORTFORMER_REPO,
        "model_installed": True,
        "runtime_installed": True,
        "installed": True,
        "reason": None,
    }


def test_sortformer_status_rejects_corrupt_model_without_exposing_path(
    monkeypatch, tmp_path
):
    model = tmp_path / "private-user-path.gguf"
    model.write_bytes(b"nope")
    monkeypatch.setattr(diarization_runtime, "sortformer_model_path", lambda: model)

    status = diarization_runtime.sortformer_status()

    assert status["model_installed"] is False
    assert status["installed"] is False
    assert status["reason"] == "Repair the installed Sortformer model bundle"
    assert str(model) not in repr(status)


# ── A Hugging Face cache entry is a symlink to an extensionless blob ─────────


def _hf_cache_entry(tmp_path, contents=b"GGUF" + b"\0" * 16):
    """``snapshots/<rev>/<dir>/<name>.gguf`` -> ``blobs/<sha256>`` (no
    extension), the way huggingface_hub lays the cache out on macOS/Linux."""
    import pytest

    blob = tmp_path / "blobs" / ("a" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(contents)
    entry = tmp_path / "snapshots" / "rev" / diarization_runtime.SORTFORMER_FILE
    entry.parent.mkdir(parents=True)
    try:
        entry.symlink_to(blob)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem cannot create symlinks")
    return entry


def _ready_runtime(monkeypatch, tmp_path):
    server = tmp_path / ("audiocpp_server.exe" if os.name == "nt" else "audiocpp_server")
    cli = tmp_path / ("audiocpp_cli.exe" if os.name == "nt" else "audiocpp_cli")
    server.write_bytes(b"server")
    cli.write_bytes(b"cli")
    cli.chmod(0o755)
    from engines.audiocpp import bootstrap

    monkeypatch.setattr(bootstrap, "resolve_server_binary", lambda: server)


def test_a_symlinked_cache_entry_is_an_installed_model(monkeypatch, tmp_path):
    """The resolved path is the extensionless blob; the model is still installed."""
    entry = _hf_cache_entry(tmp_path)
    assert entry.resolve().suffix == ""
    import huggingface_hub

    monkeypatch.delenv("OMNIVOICE_DIARIZATION_MODEL", raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: str(entry))
    assert diarization_runtime.sortformer_model_path() == entry  # named, not resolved
    _ready_runtime(monkeypatch, tmp_path)

    assert diarization_runtime.sortformer_status()["installed"] is True


def test_a_symlinked_configured_model_is_judged_by_its_name(monkeypatch, tmp_path):
    entry = _hf_cache_entry(tmp_path)
    monkeypatch.setenv("OMNIVOICE_DIARIZATION_MODEL", str(entry))
    assert diarization_runtime.sortformer_model_path() == entry
    _ready_runtime(monkeypatch, tmp_path)

    assert diarization_runtime.sortformer_status()["model_installed"] is True


def test_a_symlink_to_a_non_gguf_blob_is_broken_and_a_misnamed_file_is_missing(
    monkeypatch, tmp_path
):
    entry = _hf_cache_entry(tmp_path, contents=b"nope")
    monkeypatch.setattr(diarization_runtime, "sortformer_model_path", lambda: entry)
    assert diarization_runtime.sortformer_status()["reason"] == (
        "Repair the installed Sortformer model bundle"
    )
    misnamed = _gguf(tmp_path / "sortformer.bin")
    monkeypatch.setattr(diarization_runtime, "sortformer_model_path", lambda: misnamed)
    assert diarization_runtime.sortformer_status()["reason"] == (
        "Install the Sortformer model bundle"
    )


def test_native_sortformer_accepts_the_symlinked_entry_and_passes_its_name(
    monkeypatch, tmp_path
):
    import huggingface_hub
    from services import diarization_native

    entry = _hf_cache_entry(tmp_path)
    monkeypatch.delenv("OMNIVOICE_DIARIZATION_MODEL", raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: str(entry))
    _ready_runtime(monkeypatch, tmp_path)

    native = diarization_native.NativeSortformer()
    assert native.model == entry and native.model.suffix == ".gguf"
