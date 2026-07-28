"""
Validate no thing-on-thing pixel conflicts.

For each frame, rasterise all thing-class track polygons and check that
no pixel is covered by two thing instances at the same z_order.

A conflict is defined as:
    - Two thing polygons at the same z_order have overlapping pixels
      in the same frame.

Stuff-on-stuff and thing-on-stuff overlaps are NOT flagged here -
they are expected and resolved by Z-order rendering order.

Exits 0 on pass, 1 on any violation (CLI mode).
Returns {"passed": bool, "errors": list[str]} when called as run_checks().
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import yaml

_STUFF_LABELS = {"Water", "Sky", "Land", "Pier", "Bridge"}


def run_checks(
    xml_path: Path,
    config_path: Path,
    clips_dir: Path,
    clip_name: str,
    **_kwargs,
) -> dict:
    errors: list[str] = []

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    label_to_z: dict[str, int] = {
        lbl["name"]: lbl["z_order"] for lbl in cfg["labels"]
    }

    # Get frame size from first frame
    frames_dir = clips_dir / "frames" / clip_name
    frame_files = sorted(frames_dir.glob("*.jpeg"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        return {
            "passed": False,
            "errors": [f"No frames found in {frames_dir}"],
        }
    sample = cv2.imread(str(frame_files[0]))
    if sample is None:
        return {"passed": False, "errors": [f"Cannot read {frame_files[0]}"]}
    H, W = sample.shape[:2]
    n_frames = len(frame_files)

    # Parse XML - collect thing tracks only
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # thing_polys[frame_idx][z_order] = list of (track_id, pts_array)
    thing_polys: dict[int, dict[int, list[tuple[int, np.ndarray]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for track in root.findall(".//track"):
        label    = track.get("label", "")
        track_id = int(track.get("id", 0))
        if label in _STUFF_LABELS:
            continue
        z_order = label_to_z.get(label, -1)

        for poly in track.findall("polygon"):
            if poly.get("outside", "0") == "1":
                continue
            frame_idx = int(poly.get("frame", 0))
            pts_str   = poly.get("points", "")
            pts       = _parse_points(pts_str)
            if pts is None:
                continue
            thing_polys[frame_idx][z_order].append((track_id, pts))

    # Check each frame for same-Z thing-on-thing conflicts
    for frame_idx in range(n_frames):
        z_groups = thing_polys.get(frame_idx, {})
        for z_order, instances in z_groups.items():
            if len(instances) < 2:
                continue
            # Rasterise each instance mask separately and check pairwise overlap
            masks: list[tuple[int, np.ndarray]] = []
            for track_id, pts in instances:
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 1)
                masks.append((track_id, mask))

            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    tid_a, mask_a = masks[i]
                    tid_b, mask_b = masks[j]
                    overlap_px = int(np.logical_and(mask_a, mask_b).sum())
                    if overlap_px > 0:
                        errors.append(
                            f"frame {frame_idx:04d}: thing-on-thing conflict "
                            f"at z_order={z_order} - "
                            f"track_id={tid_a} vs track_id={tid_b} "
                            f"overlap={overlap_px} px"
                        )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
    }


def _parse_points(pts_str: str) -> np.ndarray | None:
    try:
        pairs = [p.split(",") for p in pts_str.split(";")]
        arr = np.array([[float(x), float(y)] for x, y in pairs], dtype=np.int32)
        return arr.reshape((-1, 1, 2))
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate no thing-on-thing pixel conflicts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("xml_path",    type=Path)
    p.add_argument("config_path", type=Path)
    p.add_argument("--clips-dir", type=Path,
                   default="data/interim/choosed_clips_v5-1")
    p.add_argument("--clip-name", required=True)
    p.add_argument("--report",    type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_checks(
        xml_path=args.xml_path,
        config_path=args.config_path,
        clips_dir=args.clips_dir,
        clip_name=args.clip_name,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    for err in result["errors"]:
        print(f"FAIL: {err}", file=sys.stderr)
    print("overlap_conflict_test:", "PASS" if result["passed"] else "FAIL")
    sys.exit(0 if result["passed"] else 1)
