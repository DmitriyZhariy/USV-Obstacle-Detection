from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "download_checkpoints.py"
)

_spec = importlib.util.spec_from_file_location(
    "download_checkpoints",
    _SCRIPT_PATH,
)
assert _spec is not None
assert _spec.loader is not None

downloader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(downloader)


def test_existing_checkpoint_is_not_downloaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "models" / "sam2.pt"
    output.parent.mkdir()
    output.write_bytes(b"existing-checkpoint")

    def fail_if_called(*args, **kwargs) -> None:
        raise AssertionError("urlretrieve must not run for an existing file")

    monkeypatch.setattr(
        downloader.urllib.request,
        "urlretrieve",
        fail_if_called,
    )

    result = downloader.download_checkpoint(
        "https://example.invalid/sam2.pt",
        output,
    )

    assert result == output
    assert output.read_bytes() == b"existing-checkpoint"


def test_download_uses_temporary_file_then_replaces_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "models" / "sam2.pt"
    requested: list[tuple[str, Path]] = []

    def fake_urlretrieve(url: str, destination: Path) -> None:
        requested.append((url, destination))
        destination.write_bytes(b"downloaded-checkpoint")

    monkeypatch.setattr(
        downloader.urllib.request,
        "urlretrieve",
        fake_urlretrieve,
    )

    result = downloader.download_checkpoint(
        "https://example.invalid/sam2.pt",
        output,
    )

    temporary_output = output.with_suffix(".pt.part")

    assert result == output
    assert requested == [
        ("https://example.invalid/sam2.pt", temporary_output),
    ]
    assert output.read_bytes() == b"downloaded-checkpoint"
    assert not temporary_output.exists()


def test_failed_download_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "models" / "sam2.pt"

    def failing_urlretrieve(url: str, destination: Path) -> None:
        destination.write_bytes(b"partial-download")
        raise OSError("network unavailable")

    monkeypatch.setattr(
        downloader.urllib.request,
        "urlretrieve",
        failing_urlretrieve,
    )

    with pytest.raises(RuntimeError, match="Could not download checkpoint"):
        downloader.download_checkpoint(
            "https://example.invalid/sam2.pt",
            output,
        )

    assert not output.exists()
    assert not output.with_suffix(".pt.part").exists()