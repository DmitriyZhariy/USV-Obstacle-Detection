"""
Validate >= min_coverage pixel coverage per frame.

For each frame in the clip, the union of all stuff polygons (rasterised)
must cover at least min_coverage fraction (default 0.95) of the frame area.
Thing instance polygons are excluded from this check — stuff provides the
background carpet; things sit on top.

Exits 0 on pass, 1 on any violation (CLI mode).
Returns {"passed": bool, "errors": list[str]} when called as run_checks().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np

_DEFAULT_MIN_COVERAGE = 0.95

# Z-order values for stuff classes (from design doc Z-order table).
# Tracks with z_order <= STUFF_Z_MAX are treated as stuff.
_STUFF_LABELS = {"Water", "Sky", "Land", "Pier", "Bridge"}


def run_checks(
    xml_path: Path,
    config_path: Path,
    clips_dir: Path,
    clip_name: str,
    min_coverage: float = _DEFAULT_MIN_COVERAGE,
    **_kwargs,
) -> dict:
    """
    Check pixel coverage for all frames in the clip.

    Returns
    -------
    dict with keys:
        passed : bool
        errors : list[str]   — one entry per failing frame
        stats  : dict        — per-frame coverage values (for reporting)
    """
    errors: list[str] = []
    stats:  dict[str, float] = {}

    # Load frame dimensions from first frame image
    frames_dir = clips_dir / "frames" / clip_name
    frame_files = sorted(frames_dir.glob("*.jpeg"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        return {
            "passed": False,
            "errors": [f"No frame images found in {frames_dir}"],
            "stats":  {},
        }

    sample = cv2.imread(str(frame_files[0]))
    if sample is None:
        return {
            "passed": False,
            "errors": [f"Could not read frame: {frame_files[0]}"],
            "stats":  {},
        }
    H, W = sample.shape[:2]
    total_px = H * W

    # Parse XML — collect stuff tracks only
    tree = ET.parse(xml_path)
    root = tree.getroot()
    n_frames = len(frame_files)

    # Build per-frame stuff polygon list
    # frame_polys[frame_idx] = list of polygon point arrays
    frame_polys: dict[int, list[np.ndarray]] = {i: [] for i in range(n_frames)}

    for track in root.findall(".//track"):
        label = track.get("label", "")
        if label not in _STUFF_LABELS:
            continue
        for poly in track.findall("polygon"):
            outside = poly.get("outside", "0") == "1"
            if outside:
                continue
            frame_idx = int(poly.get("frame", 0))
            pts_str = poly.get("points", "")
            if not pts_str:
                continue
            pts = _parse_points(pts_str)
            if pts is not None:
                frame_polys.setdefault(frame_idx, []).append(pts)

    # Check coverage per frame
    # Frames with no stuff polygons at all get coverage = 0.0
    for frame_idx in range(n_frames):
        polys = frame_polys.get(frame_idx, [])
        if not polys:
            coverage = 0.0
        else:
            mask = np.zeros((H, W), dtype=np.uint8)
            for pts in polys:
                cv2.fillPoly(mask, [pts], 1)
            coverage = float(mask.sum()) / total_px

        stats[f"frame_{frame_idx:04d}"] = round(coverage, 4)

        if coverage < min_coverage:
            errors.append(
                f"frame {frame_idx:04d}: stuff coverage {coverage:.1%} "
                f"< required {min_coverage:.0%}"
            )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "stats":  stats,
    }


def _parse_points(pts_str: str) -> np.ndarray | None:
    """Parse CVAT polygon points string into (N,1,2) int32 array for cv2."""
    try:
        pairs = [p.split(",") for p in pts_str.split(";")]
        arr = np.array([[float(x), float(y)] for x, y in pairs], dtype=np.int32)
        return arr.reshape((-1, 1, 2))
    except Exception:
        return None


# CLI entry point

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate stuff polygon pixel coverage per frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("xml_path",   type=Path)
    p.add_argument("config_path", type=Path)
    p.add_argument("--clips-dir", type=Path,
                   default="data/interim/choosed_clips_v5-1")
    p.add_argument("--clip-name", required=True)
    p.add_argument("--min-coverage", type=float, default=_DEFAULT_MIN_COVERAGE)
    p.add_argument("--report", type=Path, default=None,
                   help="Optional JSON report output path.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_checks(
        xml_path=args.xml_path,
        config_path=args.config_path,
        clips_dir=args.clips_dir,
        clip_name=args.clip_name,
        min_coverage=args.min_coverage,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    for err in result["errors"]:
        print(f"FAIL: {err}", file=sys.stderr)
    if result["passed"]:
        print("mask_coverage_test: PASS")
    sys.exit(0 if result["passed"] else 1)
