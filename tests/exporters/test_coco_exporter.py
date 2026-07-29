"""Unit tests for coco_exporter.py — no models required."""
import json
import pytest
from pathlib import Path
import numpy as np
import pandas as pd

from usv.auto_annotation.types import ClipData, TrackAnnotation, PolygonKeyframe
from usv.auto_annotation.exporters.coco_exporter import export_coco, COCO_CATEGORIES


def _make_clip(n_frames: int = 10, h: int = 480, w: int = 640) -> ClipData:
    return ClipData(
        clip_name="test_clip",
        frames=[np.zeros((h, w, 3), dtype=np.uint8)] * n_frames,
        keyframe_idx=n_frames // 2,
        n_frames=n_frames,
        frame_height=h,
        frame_width=w,
        meta_df=pd.DataFrame(),
        original_video="test.avi",
    )


def _make_track(
    track_id: int,
    label: str,
    keyframes: list[PolygonKeyframe],
) -> TrackAnnotation:
    return TrackAnnotation(
        track_id=track_id,
        label=label,
        z_order=30,
        keyframes=keyframes,
    )


# ── тест 1: базовая структура COCO JSON ──────────────────────────────────────

def test_coco_json_structure(tmp_path):
    clip = _make_clip()
    kf = PolygonKeyframe(
        frame_idx=5,
        points=[(10.0, 20.0), (50.0, 20.0), (50.0, 60.0), (10.0, 60.0)],
        keyframe=True,
        outside=False,
    )
    tracks = [_make_track(1, "Vessel", [kf])]
    out = export_coco(clip, tracks, tmp_path)

    data = json.loads(out.read_text())
    assert "images" in data
    assert "annotations" in data
    assert "categories" in data
    assert len(data["images"]) == clip.n_frames     # один entry на каждый кадр
    assert len(data["annotations"]) == 1


# ── тест 2: координаты полигона сохраняются без искажений ────────────────────

def test_polygon_coordinates_preserved(tmp_path):
    pts = [(10.5, 20.5), (50.0, 20.5), (50.0, 60.0), (10.5, 60.0)]
    kf = PolygonKeyframe(frame_idx=3, points=pts, keyframe=True, outside=False)
    tracks = [_make_track(1, "Buoy", [kf])]
    out = export_coco(_make_clip(), tracks, tmp_path)

    ann = json.loads(out.read_text())["annotations"][0]
    flat = ann["segmentation"][0]
    # Восстанавливаем точки из flat list
    recovered = [(flat[i], flat[i+1]) for i in range(0, len(flat), 2)]
    assert recovered == pts


# ── тест 3: outside=True keyframes НЕ попадают в аннотации ───────────────────

def test_outside_keyframes_excluded(tmp_path):
    kf_visible = PolygonKeyframe(
        frame_idx=2,
        points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        keyframe=True, outside=False,
    )
    kf_outside = PolygonKeyframe(
        frame_idx=7,
        points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        keyframe=True, outside=True,   # <── должен быть отфильтрован
    )
    tracks = [_make_track(1, "Vessel", [kf_visible, kf_outside])]
    out = export_coco(_make_clip(), tracks, tmp_path)

    anns = json.loads(out.read_text())["annotations"]
    assert len(anns) == 1
    assert anns[0]["image_id"] == 2   # только visible keyframe


# ── тест 4: stuff-классы (Sky, Water) не попадают в аннотации ────────────────

def test_stuff_labels_skipped(tmp_path):
    kf = PolygonKeyframe(
        frame_idx=5,
        points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        keyframe=True, outside=False,
    )
    # Water — stuff-класс, нет в THING_LABELS
    tracks = [_make_track(1000, "Water", [kf])]
    out = export_coco(_make_clip(), tracks, tmp_path)

    anns = json.loads(out.read_text())["annotations"]
    assert len(anns) == 0   # Water должен быть проигнорирован


# ── тест 5: bbox вычисляется корректно из полигона ───────────────────────────

def test_bbox_computed_from_polygon(tmp_path):
    # Прямоугольник: x=[10,50], y=[20,60] → bbox=[10,20,40,40]
    pts = [(10.0, 20.0), (50.0, 20.0), (50.0, 60.0), (10.0, 60.0)]
    kf = PolygonKeyframe(frame_idx=0, points=pts, keyframe=True, outside=False)
    tracks = [_make_track(2, "Buoy", [kf])]
    out = export_coco(_make_clip(), tracks, tmp_path)

    ann = json.loads(out.read_text())["annotations"][0]
    x, y, w, h = ann["bbox"]
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(20.0)
    assert w == pytest.approx(40.0)   # 50 - 10
    assert h == pytest.approx(40.0)   # 60 - 20


# ── тест 6: несколько треков → уникальные annotation id ──────────────────────

def test_annotation_ids_are_unique(tmp_path):
    def _kf(frame_idx: int):
        return PolygonKeyframe(
            frame_idx=frame_idx,
            points=[(0.0,0.0),(10.0,0.0),(10.0,10.0),(0.0,10.0)],
            keyframe=True, outside=False,
        )
    tracks = [
        _make_track(1, "Vessel", [_kf(0), _kf(3), _kf(7)]),
        _make_track(2, "Buoy",   [_kf(1), _kf(5)]),
    ]
    out = export_coco(_make_clip(n_frames=10), tracks, tmp_path)

    anns = json.loads(out.read_text())["annotations"]
    ids = [a["id"] for a in anns]
    assert len(ids) == len(set(ids))   # все id уникальны
    assert len(anns) == 5              # 3 + 2 keyframes


# ── тест 7: images-список соответствует n_frames ─────────────────────────────

def test_images_count_matches_n_frames(tmp_path):
    clip = _make_clip(n_frames=25)
    out = export_coco(clip, [], tmp_path)
    data = json.loads(out.read_text())
    assert len(data["images"]) == 25
    # file_name формат: 00000.jpg ... 00024.jpg
    assert data["images"][0]["file_name"] == "00000.jpg"
    assert data["images"][-1]["file_name"] == "00024.jpg"


# ── тест 8: track_id проставляется в аннотацию ───────────────────────────────

def test_track_id_in_annotation(tmp_path):
    kf = PolygonKeyframe(
        frame_idx=1,
        points=[(0.0,0.0),(5.0,0.0),(5.0,5.0),(0.0,5.0)],
        keyframe=True, outside=False,
    )
    tracks = [_make_track(42, "Vessel", [kf])]
    out = export_coco(_make_clip(), tracks, tmp_path)

    ann = json.loads(out.read_text())["annotations"][0]
    assert ann["track_id"] == 42