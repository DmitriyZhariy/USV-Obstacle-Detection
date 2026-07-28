"""
Export existing pipeline predictions to a CVAT-importable zip.

Reads the annotations.xml already produced by pipeline.py and re-packages
it into a fresh zip (useful after manual XML edits or format fixes).
Optionally uploads to a live CVAT instance via REST API.

Usage:
    python -m scripts.export_cvat `
        --annotations-dir data/interim/auto_annotations/cvat_export `
        --clip-name left_MOVI0017_0001

    # With CVAT upload:
    python -m scripts.export_cvat `
        --annotations-dir data/interim/auto_annotations/cvat_export `
        --clip-name left_MOVI0017_0001 `
        --upload-to-cvat `
        --cvat-url http://10.49.234.230:8080 `
        --cvat-project-id 3
"""
from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class CVATExporter:

    def __init__(self, args: argparse.Namespace) -> None:
        self.annotations_dir = Path(args.annotations_dir)
        self.upload           = args.upload_to_cvat
        self.cvat_url         = args.cvat_url
        self.cvat_project_id  = args.cvat_project_id
        self.skip_existing    = args.skip_existing

    def run(self, clip_name: str | None, process_all: bool) -> None:
        clips = self._resolve_clips(clip_name, process_all)
        logger.info("Clips to export: %d", len(clips))
        for name in clips:
            try:
                self._export_one(name)
            except Exception:
                logger.exception("[FAIL] %s", name)

    def _resolve_clips(
        self, clip_name: str | None, process_all: bool
    ) -> list[str]:
        if process_all:
            return sorted(
                p.stem for p in self.annotations_dir.glob("*.xml")
                if not p.stem.endswith("_manifest")
            )
        if clip_name:
            return [clip_name]
        logger.error("Specify --clip-name or --all")
        sys.exit(1)

    def _export_one(self, clip_name: str) -> None:
        xml_path = self.annotations_dir / f"{clip_name}.xml"
        zip_path = self.annotations_dir / f"{clip_name}.zip"

        if not xml_path.exists():
            logger.error("XML not found: %s", xml_path)
            return

        if self.skip_existing and zip_path.exists():
            logger.info("[SKIP] %s — zip already exists", clip_name)
        else:
            # Re-package XML into zip
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(xml_path, arcname="annotations.xml")
            logger.info("[EXPORT] %s → %s", clip_name, zip_path)

        if self.upload:
            self._upload(clip_name, zip_path)

    def _upload(self, clip_name: str, zip_path: Path) -> None:
        """
        Upload zip to CVAT via REST API.
        Requires: pip install requests
        CVAT task must already exist as a Video task (not Image Collection).
        """
        try:
            import requests
        except ImportError:
            logger.error(
                "Upload requires: pip install requests. Skipping upload."
            )
            return

        if not self.cvat_url or not self.cvat_project_id:
            logger.error(
                "Provide --cvat-url and --cvat-project-id for upload. "
                "Skipping %s.", clip_name,
            )
            return

        # Find task by name (clip_name)
        tasks_url = f"{self.cvat_url.rstrip('/')}/api/tasks"
        resp = requests.get(tasks_url, params={"name": clip_name}, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])

        if not results:
            logger.warning(
                "No CVAT task found with name '%s'. "
                "Create the task manually first, then re-run with --upload-to-cvat.",
                clip_name,
            )
            return

        task_id = results[0]["id"]
        upload_url = f"{tasks_url}/{task_id}/annotations/"

        with open(zip_path, "rb") as f:
            upload_resp = requests.put(
                upload_url,
                files={"annotation_file": (zip_path.name, f, "application/zip")},
                params={"format": "CVAT for video 1.1"},
                timeout=60,
            )

        if upload_resp.status_code in (200, 201, 202):
            logger.info(
                "[UPLOAD] %s → CVAT task_id=%d  status=%d",
                clip_name, task_id, upload_resp.status_code,
            )
        else:
            logger.error(
                "[UPLOAD FAIL] %s  status=%d  body=%s",
                clip_name, upload_resp.status_code, upload_resp.text[:200],
            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-package CVAT XML into import zip; optionally upload.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--annotations-dir",
        default="data/interim/auto_annotations/cvat_export",
    )
    p.add_argument("--clip-name",  default=None)
    p.add_argument("--all",        action="store_true", dest="all_clips")
    p.add_argument("--upload-to-cvat", action="store_true", default=False)
    p.add_argument("--cvat-url",       default=None)
    p.add_argument("--cvat-project-id", default=None, type=int)
    p.add_argument("--skip-existing",    action="store_true",  default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exporter = CVATExporter(args)
    exporter.run(clip_name=args.clip_name, process_all=args.all_clips)
