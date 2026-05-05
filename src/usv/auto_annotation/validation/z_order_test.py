"""
Validates that every <track> z_order in a CVAT XML matches the label definition
in configs/auto_annotation.yaml.

Exit 0: all z_order values are correct.
Exit 1: any mismatch found (prints violations table).
"""
import argparse
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml


def load_expected_z_orders(config_path: Path) -> dict[str, int]:
    """Build {label_name: z_order} from auto_annotation.yaml."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {label["name"]: label["z_order"] for label in cfg["labels"]}


def validate_z_orders(xml_path: Path, expected: dict[str, int]) -> list[dict]:
    """Parse CVAT XML and return list of z_order violations."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    violations = []

    for track_el in root.findall("track"):
        track_id = track_el.get("id", "?")
        label = track_el.get("label", "")
        actual_z = track_el.get("z_order")

        if label not in expected:
            violations.append({
                "track_id": track_id,
                "label": label,
                "expected": "N/A (unknown label)",
                "actual": actual_z,
                "error": "unknown_label",
            })
            continue

        expected_z = expected[label]
        if actual_z is None or int(actual_z) != expected_z:
            violations.append({
                "track_id": track_id,
                "label": label,
                "expected": expected_z,
                "actual": actual_z,
                "error": "z_order_mismatch",
            })

    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate z_order attributes in a CVAT Video XML against the config."
    )
    parser.add_argument(
        "--xml-path", required=True, type=Path,
        help="Path to the CVAT annotations.xml file to validate.",
    )
    parser.add_argument(
        "--config", default=Path("configs/auto_annotation.yaml"), type=Path,
        help="Path to auto_annotation.yaml (default: configs/auto_annotation.yaml).",
    )
    parser.add_argument(
        "--report-path", type=Path, default=None,
        help="Optional: write JSON violation report to this path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.xml_path.exists():
        print(f"ERROR: XML file not found: {args.xml_path}", file=sys.stderr)
        sys.exit(1)
    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    expected = load_expected_z_orders(args.config)
    violations = validate_z_orders(args.xml_path, expected)

    report = {
        "xml_path": str(args.xml_path),
        "config_path": str(args.config),
        "violations": violations,
        "passed": len(violations) == 0,
    }

    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if violations:
        print(f"FAIL: {len(violations)} z_order violation(s) found:")
        for v in violations:
            print(
                f"  track_id={v['track_id']} label={v['label']} "
                f"expected={v['expected']} actual={v['actual']}"
            )
        sys.exit(1)
    else:
        print(f"PASS: all {len(expected)} label z_order values correct in {args.xml_path.name}")
        sys.exit(0)


if __name__ == "__main__":
    main()