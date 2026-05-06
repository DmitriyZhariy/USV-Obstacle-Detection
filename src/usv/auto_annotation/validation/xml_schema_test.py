"""
Structural validator for CVAT 1.1 Video XML exports.
Exits 0 on pass, exits 1 on any violation.
Complements z_order_test.py (content validation) with schema validation.
"""
from __future__ import annotations
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_REQUIRED_TRACK_ATTRS = {"id", "label", "z_order"}
_REQUIRED_POLYGON_ATTRS = {"frame", "points", "outside", "occluded", "keyframe"}


def validate_xml(xml_path: Path) -> list[str]:
    violations: list[str] = []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        return [f"XML parse error: {e}"]

    root = tree.getroot()
    if root.tag != "annotations":
        violations.append(f"Root tag is '{root.tag}', expected 'annotations'")

    seen_ids: set[str] = set()
    for track in root.findall("track"):
        for attr in _REQUIRED_TRACK_ATTRS:
            if attr not in track.attrib:
                violations.append(f"<track> missing attribute '{attr}'")

        tid = track.get("id", "?")
        if tid in seen_ids:
            violations.append(f"Duplicate track id='{tid}'")
        seen_ids.add(tid)

        polygons = track.findall("polygon")
        if not polygons:
            violations.append(f"Track id={tid} has no <polygon> children")
            continue

        if not any(p.get("keyframe") == "1" for p in polygons):
            violations.append(f"Track id={tid} has no polygon with keyframe='1'")

        if all(p.get("outside") == "1" for p in polygons):
            violations.append(
                f"Track id={tid}: all polygons have outside='1' (empty track)"
            )

        for poly in polygons:
            for attr in _REQUIRED_POLYGON_ATTRS:
                if attr not in poly.attrib:
                    violations.append(
                        f"Track id={tid} polygon frame={poly.get('frame','?')}: "
                        f"missing attribute '{attr}'"
                    )

    return violations


def main() -> None:
    p = argparse.ArgumentParser(description="Validate CVAT 1.1 Video XML schema.")
    p.add_argument("xml_path", help="Path to annotations.xml")
    p.add_argument("--report", default=None, help="Optional JSON report output path.")
    args = p.parse_args()

    xml_path = Path(args.xml_path)
    if not xml_path.exists():
        print(f"ERROR: File not found: {xml_path}", file=sys.stderr)
        sys.exit(1)

    violations = validate_xml(xml_path)
    report = {
        "xml_path": str(xml_path),
        "violations": violations,
        "passed": not violations,
    }
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    if violations:
        print(f"FAIL — {len(violations)} violation(s):")
        for v in violations:
            print(f"  • {v}")
        sys.exit(1)

    print("PASS — XML schema valid.")
    sys.exit(0)


if __name__ == "__main__":
    main()
