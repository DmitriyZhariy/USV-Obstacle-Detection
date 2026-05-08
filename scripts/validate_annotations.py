"""
Run all validation checks on auto-annotation outputs.

Aggregates results from:
    - validation/xml_schema_test.py     - CVAT 1.1 XML structure
    - validation/z_order_test.py        - Z-order per label from config
    - validation/mask_coverage_test.py  - at least min_coverage pixel coverage
    - validation/track_consistency_test.py - track continuity + outside flags
    - validation/overlap_conflict_test.py  - no thing-on-thing pixel conflicts

Exits 0 if all checks pass, 1 if any check fails.
Report written to --report-path as JSON.

Usage:
    python -m scripts.validate_annotations `
        --annotations-dir data/interim/auto_annotations `
        --clips-dir data/interim/choosed_clips_v5-1 `
        --config configs/auto_annotation.yaml `
        --clip-name left_MOVI0017_0001 `
        --report-path reports/auto_annotation/validation.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Registry of validation modules.
# Each must expose: run_checks(xml_path, config_path, clips_dir, **kwargs)
# dict {"passed": bool, "errors": list[str]}
_VALIDATORS = [
    "usv.auto_annotation.validation.xml_schema_test",
    "usv.auto_annotation.validation.z_order_test",
    "usv.auto_annotation.validation.mask_coverage_test",
    "usv.auto_annotation.validation.track_consistency_test",
    "usv.auto_annotation.validation.overlap_conflict_test",
]


class ValidationRunner:

    def __init__(self, args: argparse.Namespace) -> None:
        self.annotations_dir = Path(args.annotations_dir)
        self.clips_dir       = Path(args.clips_dir)
        self.config_path     = Path(args.config)
        self.report_path     = Path(args.report_path)
        self.min_coverage    = args.min_coverage
        self.checks          = args.checks   # "all" or comma-separated names

    def run(self, clip_name: str | None, process_all: bool) -> int:
        clips = self._resolve_clips(clip_name, process_all)
        logger.info("Clips to validate: %d", len(clips))

        all_results: dict[str, dict] = {}
        any_failed = False

        for name in clips:
            result = self._validate_one(name)
            all_results[name] = result
            if not result["passed"]:
                any_failed = True

        self._write_report(all_results)
        return 1 if any_failed else 0

    def _resolve_clips(
        self, clip_name: str | None, process_all: bool
    ) -> list[str]:
        if process_all:
            return sorted(
                p.stem for p in
                (self.annotations_dir / "cvat_export").glob("*.xml")
                if not p.stem.endswith("_manifest")
            )
        if clip_name:
            return [clip_name]
        logger.error("Specify --clip-name or --all")
        sys.exit(1)

    def _validate_one(self, clip_name: str) -> dict:
        xml_path = (
            self.annotations_dir / "cvat_export" / f"{clip_name}.xml"
        )
        if not xml_path.exists():
            logger.error("XML not found: %s", xml_path)
            return {"passed": False, "errors": [f"XML not found: {xml_path}"]}

        clip_results: dict[str, dict] = {}
        passed_all = True
        t0 = time.perf_counter()

        for module_path in _VALIDATORS:
            check_name = module_path.split(".")[-1]
            if self.checks != "all" and check_name not in self.checks.split(","):
                continue
            try:
                mod = importlib.import_module(module_path)
                result = mod.run_checks(
                    xml_path=xml_path,
                    config_path=self.config_path,
                    clips_dir=self.clips_dir,
                    clip_name=clip_name,
                    min_coverage=self.min_coverage,
                )
            except Exception as exc:
                result = {"passed": False, "errors": [str(exc)]}
                logger.exception("Validator %s raised an exception.", check_name)

            clip_results[check_name] = result
            status = "PASS" if result["passed"] else "FAIL"
            logger.info(
                "  [%s] %s / %s - %s",
                status, clip_name, check_name,
                "; ".join(result.get("errors", [])) or "ok",
            )
            if not result["passed"]:
                passed_all = False

        elapsed = time.perf_counter() - t0
        return {
            "passed":  passed_all,
            "checks":  clip_results,
            "elapsed_s": round(elapsed, 2),
        }

    def _write_report(self, results: dict) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        n_clips   = len(results)
        n_failed  = sum(1 for r in results.values() if not r["passed"])
        logger.info(
            "Report written: %s  (%d/%d clips passed)",
            self.report_path, n_clips - n_failed, n_clips,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all validation checks on auto-annotation outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotations-dir", default="data/interim/auto_annotations")
    p.add_argument("--clips-dir",       default="data/interim/choosed_clips_v5-1")
    p.add_argument("--config",          default="configs/auto_annotation.yaml")
    p.add_argument("--clip-name",       default=None)
    p.add_argument("--all",             action="store_true", dest="all_clips")
    p.add_argument(
        "--checks", default="all",
        help="Comma-separated check names, or 'all'. "
             "Names: xml_schema_test,z_order_test,mask_coverage_test,"
             "track_consistency_test,overlap_conflict_test",
    )
    p.add_argument("--min-coverage", default=0.95, type=float,
                   help="Minimum pixel coverage fraction (mask_coverage_test).")
    p.add_argument(
        "--report-path",
        default="reports/auto_annotation/validation.json",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    runner = ValidationRunner(args)
    exit_code = runner.run(
        clip_name=args.clip_name,
        process_all=args.all_clips,
    )
    sys.exit(exit_code)
