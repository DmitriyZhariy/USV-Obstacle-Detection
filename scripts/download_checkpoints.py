"""
Download the SAM 2.1 Hiera Small checkpoint required by cpu-sam2 mode.

Examples:
    uv run python -m scripts.download_checkpoints
    uv run python -m scripts.download_checkpoints --output models/sam2.1_hiera_small.pt
    uv run python -m scripts.download_checkpoints --force
"""
from __future__ import annotations

import argparse
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path

SAM2_SMALL_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/"
    "092824/sam2.1_hiera_small.pt"
)
DEFAULT_OUTPUT = Path("models/sam2.1_hiera_small.pt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_checkpoint(
    url: str,
    output: Path,
    *,
    force: bool = False,
) -> Path:
    """Download a checkpoint unless it already exists."""
    if output.exists() and not force:
        logger.info(
            "Checkpoint already exists: %s (%d bytes). Use --force to re-download.",
            output,
            output.stat().st_size,
        )
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(f"{output.suffix}.part")

    try:
        logger.info("Downloading SAM 2.1 Hiera Small checkpoint...")
        urllib.request.urlretrieve(url, temporary_output)
        temporary_output.replace(output)
    except (urllib.error.URLError, OSError) as exc:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download checkpoint from {url}: {exc}"
        ) from exc

    logger.info(
        "Saved checkpoint: %s (%d bytes)",
        output,
        output.stat().st_size,
    )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the SAM 2.1 Hiera Small checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help="Destination checkpoint path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when the output file already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        download_checkpoint(
            SAM2_SMALL_URL,
            args.output,
            force=args.force,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())