"""
Serializes TrackAnnotation objects into CVAT 1.1 Video XML format.
All tracks use Track mode - required by the annotation manual.
"""
from __future__ import annotations
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

from usv.auto_annotation.types import ClipData, TrackAnnotation


# Label list must match CVAT task definition.
LABEL_NAMES: list[str] = [
    "Water", "Sky", "Land", "Pier", "Bridge",
    "Vessel", "LandingMark", "BridgeLight", "Buoy", "Other", "Void",
]


def _fmt_points(points: list[tuple[float, float]]) -> str:
    """Convert [(x0,y0), ...] to CVAT points string 'x0,y0;x1,y1;...'"""
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _bool_attr(value: bool) -> str:
    return "1" if value else "0"


def build_xml(
    clip_data: ClipData,
    tracks: list[TrackAnnotation],
    label_names: list[str] | None = None,
) -> ET.Element:
    """Build the CVAT 1.1 XML element tree in memory.

    Args:
        clip_data:    ClipData for the clip being exported.
        tracks:       List of TrackAnnotation objects to serialize.
        label_names:  Override the default label list (useful for testing).

    Returns:
        ET.Element root <annotations> element.

    Raises:
        ValueError: If a track has an unknown label or duplicate track IDs exist.
    """
    labels = label_names if label_names is not None else LABEL_NAMES
    label_set = set(labels)

    # Validate inputs before writing anything
    seen_ids: set[int] = set()
    for track in tracks:
        if track.label not in label_set:
            raise ValueError(
                f"Track {track.track_id} has unknown label '{track.label}'. "
                f"Valid labels: {sorted(label_set)}"
            )
        if track.track_id in seen_ids:
            raise ValueError(f"Duplicate track_id: {track.track_id}")
        seen_ids.add(track.track_id)
        if not track.keyframes:
            raise ValueError(
                f"Track {track.track_id} ({track.label}) has no keyframes."
            )
        if not any(kf.keyframe for kf in track.keyframes):
            raise ValueError(
                f"Track {track.track_id} ({track.label}) has no frame "
                f"with keyframe=True. CVAT requires at least one keyframe per track."
            )
        if not any(not kf.outside for kf in track.keyframes):
            raise ValueError(
                f"Track {track.track_id} ({track.label}) has no visible (non-outside) "
                f"keyframe. CVAT requires at least one visible keyframe per track."
            )

    # Root
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    # meta
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = clip_data.clip_name
    ET.SubElement(task, "size").text = str(clip_data.n_frames)

    labels_el = ET.SubElement(task, "labels")
    for name in labels:
        label_el = ET.SubElement(labels_el, "label")
        ET.SubElement(label_el, "name").text = name
        ET.SubElement(label_el, "attributes")

    # track elements
    for track in sorted(tracks, key=lambda t: t.track_id):
        track_el = ET.SubElement(root, "track")
        track_el.set("id", str(track.track_id))
        track_el.set("label", track.label)
        track_el.set("z_order", str(track.z_order))
        track_el.set("source", track.source)

        for kf in sorted(track.keyframes, key=lambda k: k.frame_idx):
            poly_el = ET.SubElement(track_el, "polygon")
            poly_el.set("frame", str(kf.frame_idx))
            poly_el.set("points", _fmt_points(kf.points))
            poly_el.set("outside", _bool_attr(kf.outside))
            poly_el.set("occluded", _bool_attr(kf.occluded))
            poly_el.set("keyframe", _bool_attr(kf.keyframe))
            poly_el.set("z_order", str(track.z_order))

    return root


def _pretty_xml(root: ET.Element) -> str:
    """Return indented XML string with declaration."""
    raw = ET.tostring(root, encoding="unicode")
    parsed = minidom.parseString(raw)
    return parsed.toprettyxml(indent="  ", encoding=None)


def export_clip(
    clip_data: ClipData,
    tracks: list[TrackAnnotation],
    output_dir: Path,
    label_names: list[str] | None = None,
    write_zip: bool = True,
) -> Path:
    """Serialize tracks to CVAT XML and optionally package into a zip.

    Args:
        clip_data:   ClipData for the clip.
        tracks:      Track annotations to export.
        output_dir:  Directory where {clip_name}/ folder is created.
        label_names: Optional label override.
        write_zip:   If True, also writes {clip_name}.zip for CVAT import.

    Returns:
        Path to the written annotations.xml file.
    """
    clip_dir = output_dir / clip_data.clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    xml_path = clip_dir / "annotations.xml"
    root = build_xml(clip_data, tracks, label_names=label_names)
    xml_string = _pretty_xml(root)
    xml_path.write_text(xml_string, encoding="utf-8")

    if write_zip:
        zip_path = output_dir / f"{clip_data.clip_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(xml_path, arcname="annotations.xml")

    return xml_path
