"""
Validate track continuity and outside flags.

Checks:
    1. Every track has at least one visible (outside=False) keyframe.
    2. No track has keyframes after an outside=True frame
       (track must be retired on outside=True — no re-appearance).
    3. Every track's keyframe frame indices are strictly ascending.
    4. No duplicate track IDs in the XML.
    5. Every track has a label matching a known config label.

Exits 0 on pass, 1 on any violation (CLI mode).
Returns {"passed": bool, "errors": list[str]} when called as run_checks().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml


def run_checks(
    xml_path: Path,
    config_path: Path,
    **_kwargs,
) -> dict:
    errors: list[str] = []

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    known_labels = {lbl["name"] for lbl in cfg["labels"]}

    tree = ET.parse(xml_path)
    root = tree.getroot()

    seen_track_ids: set[int] = set()

    for track in root.findall(".//track"):
        track_id_str = track.get("id", "?")
        label        = track.get("label", "")
        try:
            track_id = int(track_id_str)
        except ValueError:
            errors.append(f"track id='{track_id_str}': non-integer ID")
            continue

        # check duplicate IDs
        if track_id in seen_track_ids:
            errors.append(f"track id={track_id}: duplicate track ID")
        seen_track_ids.add(track_id)

        # check unknown label
        if label not in known_labels:
            errors.append(
                f"track id={track_id}: unknown label '{label}'"
            )

        polys = track.findall("polygon")
        if not polys:
            errors.append(f"track id={track_id} ('{label}'): no polygon elements")
            continue

        frame_indices  = [int(p.get("frame", 0)) for p in polys]
        outside_flags  = [p.get("outside", "0") == "1" for p in polys]

        # check strictly ascending frame indices
        for i in range(1, len(frame_indices)):
            if frame_indices[i] <= frame_indices[i - 1]:
                errors.append(
                    f"track id={track_id} ('{label}'): frame indices not "
                    f"strictly ascending at positions {i-1},{i}: "
                    f"{frame_indices[i-1]}, {frame_indices[i]}"
                )

        # check at least one visible keyframe
        has_visible = any(not o for o in outside_flags)
        if not has_visible:
            errors.append(
                f"track id={track_id} ('{label}'): "
                f"no visible (outside=False) keyframe"
            )

        # check no keyframes after outside=True (retirement rule)
        outside_positions = [i for i, o in enumerate(outside_flags) if o]
        if outside_positions:
            last_outside = outside_positions[-1]
            if last_outside < len(outside_flags) - 1:
                errors.append(
                    f"track id={track_id} ('{label}'): "
                    f"keyframes found after outside=True at "
                    f"frame {frame_indices[last_outside]} — "
                    f"track must be retired on outside=True"
                )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate track continuity and outside flags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("xml_path",    type=Path)
    p.add_argument("config_path", type=Path)
    p.add_argument("--report",    type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_checks(
        xml_path=args.xml_path,
        config_path=args.config_path,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    for err in result["errors"]:
        print(f"FAIL: {err}", file=sys.stderr)
    print("track_consistency_test:", "PASS" if result["passed"] else "FAIL")
    sys.exit(0 if result["passed"] else 1)
