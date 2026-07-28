"""Tests for cvat_xml_exporter.py - no model inference, synthetic data only."""
import pytest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from usv.auto_annotation.exporters.cvat_xml_exporter import (
    build_xml,
    export_clip,
    LABEL_NAMES,
)
from usv.auto_annotation.types import (
    ClipData,
    PolygonKeyframe,
    TrackAnnotation,
)
import numpy as np
import pandas as pd


# Fixtures

@pytest.fixture
def minimal_clip() -> ClipData:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    return ClipData(
        clip_name="test_left_FIXTURE_0001",
        frames=[frame] * 3,
        keyframe_idx=1,
        n_frames=3,
        frame_height=64,
        frame_width=64,
        meta_df=pd.DataFrame({
            "local_seq_idx": [1, 2, 3],
            "original_video": ["FIXTURE.avi"] * 3,
            "original_frame_idx": [100, 106, 112],
        }),
        original_video="FIXTURE.avi",
    )


@pytest.fixture
def vessel_track() -> TrackAnnotation:
    return TrackAnnotation(
        track_id=1,
        label="Vessel",
        z_order=30,
        keyframes=[
            PolygonKeyframe(frame_idx=0, points=[(10, 10), (50, 10), (50, 40), (10, 40)]),
            PolygonKeyframe(frame_idx=2, points=[(12, 11), (52, 11), (52, 41), (12, 41)]),
        ],
    )


@pytest.fixture
def water_track() -> TrackAnnotation:
    return TrackAnnotation(
        track_id=0,
        label="Water",
        z_order=0,
        keyframes=[
            PolygonKeyframe(frame_idx=0, points=[(0, 32), (64, 32), (64, 64), (0, 64)]),
        ],
    )


# Structure tests

def test_xml_has_version(minimal_clip, vessel_track):
    root = build_xml(minimal_clip, [vessel_track])
    assert root.find("version").text == "1.1"


def test_xml_has_task_name(minimal_clip, vessel_track):
    root = build_xml(minimal_clip, [vessel_track])
    assert root.find("meta/task/name").text == minimal_clip.clip_name


def test_xml_has_all_labels(minimal_clip, vessel_track):
    root = build_xml(minimal_clip, [vessel_track])
    label_names = [el.find("name").text for el in root.findall("meta/task/labels/label")]
    for expected in LABEL_NAMES:
        assert expected in label_names


# Track attribute tests

def test_track_z_order(minimal_clip, vessel_track):
    root = build_xml(minimal_clip, [vessel_track])
    track_el = root.find("track[@label='Vessel']")
    assert track_el is not None
    assert track_el.get("z_order") == "30"


def test_track_source_is_auto(minimal_clip, vessel_track):
    root = build_xml(minimal_clip, [vessel_track])
    track_el = root.find("track[@label='Vessel']")
    assert track_el.get("source") == "auto"


def test_polygon_keyframe_attr(minimal_clip, vessel_track):
    root = build_xml(minimal_clip, [vessel_track])
    polygons = root.find("track[@label='Vessel']").findall("polygon")
    # Both are keyframe=True by default
    for poly in polygons:
        assert poly.get("keyframe") == "1"


def test_polygon_occluded_is_zero(minimal_clip, vessel_track):
    root = build_xml(minimal_clip, [vessel_track])
    for poly in root.find("track[@label='Vessel']").findall("polygon"):
        assert poly.get("occluded") == "0"


def test_polygon_outside_flag(minimal_clip):
    """A track whose last keyframe has outside=True."""
    track = TrackAnnotation(
        track_id=5,
        label="Buoy",
        z_order=50,
        keyframes=[
            PolygonKeyframe(frame_idx=0, points=[(5, 5), (15, 5), (15, 15), (5, 15)]),
            PolygonKeyframe(frame_idx=2, points=[(5, 5), (15, 5), (15, 15), (5, 15)],
                            outside=True),
        ],
    )
    root = build_xml(minimal_clip, [track])
    polys = root.find("track[@label='Buoy']").findall("polygon")
    outside_values = [p.get("outside") for p in polys]
    assert "1" in outside_values


def test_no_duplicate_track_ids(minimal_clip, vessel_track, water_track):
    root = build_xml(minimal_clip, [vessel_track, water_track])
    ids = [t.get("id") for t in root.findall("track")]
    assert len(ids) == len(set(ids))


def test_points_format(minimal_clip, vessel_track):
    """Points must be serialized as 'x,y;x,y;...' with no trailing semicolon."""
    root = build_xml(minimal_clip, [vessel_track])
    poly = root.find("track[@label='Vessel']/polygon")
    points_str = poly.get("points")
    assert ";" in points_str
    assert not points_str.endswith(";")
    pairs = points_str.split(";")
    for pair in pairs:
        x, y = pair.split(",")
        float(x)
        float(y)


# Validation / error tests

def test_unknown_label_raises(minimal_clip):
    bad_track = TrackAnnotation(
        track_id=99,
        label="UNKNOWN_CLASS",
        z_order=30,
        keyframes=[PolygonKeyframe(frame_idx=0, points=[(0, 0), (10, 0), (10, 10), (0, 10)])],
    )
    with pytest.raises(ValueError, match="unknown label"):
        build_xml(minimal_clip, [bad_track])


def test_duplicate_track_id_raises(minimal_clip, vessel_track):
    duplicate = TrackAnnotation(
        track_id=1,  # same as vessel_track
        label="Buoy",
        z_order=50,
        keyframes=[PolygonKeyframe(frame_idx=0, points=[(0, 0), (10, 0), (10, 10), (0, 10)])],
    )
    with pytest.raises(ValueError, match="Duplicate track_id"):
        build_xml(minimal_clip, [vessel_track, duplicate])


def test_empty_keyframes_raises(minimal_clip):
    bad_track = TrackAnnotation(track_id=3, label="Land", z_order=5, keyframes=[])
    with pytest.raises(ValueError, match="no keyframes"):
        build_xml(minimal_clip, [bad_track])


# File output tests

def test_export_writes_xml(tmp_path, minimal_clip, vessel_track, water_track):
    xml_path = export_clip(minimal_clip, [vessel_track, water_track], tmp_path)
    assert xml_path.exists()
    tree = ET.parse(xml_path)
    assert tree.getroot().tag == "annotations"


def test_export_writes_zip(tmp_path, minimal_clip, vessel_track):
    export_clip(minimal_clip, [vessel_track], tmp_path, write_zip=True)
    zip_path = tmp_path / f"{minimal_clip.clip_name}.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        assert "annotations.xml" in zf.namelist()


def test_export_xml_is_valid_utf8(tmp_path, minimal_clip, vessel_track):
    xml_path = export_clip(minimal_clip, [vessel_track], tmp_path)
    content = xml_path.read_text(encoding="utf-8")
    assert "<?xml" in content