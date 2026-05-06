import numpy as np
import pytest
from usv.auto_annotation.tracker.iou_tracker import IoUTracker


def _det(label="Vessel", bbox=(10, 10, 50, 50), z_order=30, conf=0.9):
    return {
        "label": label,
        "class_id": 5,
        "z_order": z_order,
        "confidence": conf,
        "bbox_xyxy": bbox,
        "mask": None,
    }


# basic lifecycle

def test_single_object_tracked_across_frames():
    t = IoUTracker()
    for i in range(5):
        t.update(i, [_det(bbox=(10+i, 10, 50+i, 50))])
    tracks = t.finalize()
    assert len(tracks) == 1
    assert tracks[0].label == "Vessel"
    assert len(tracks[0].keyframes) == 5
    assert all(kf.keyframe for kf in tracks[0].keyframes)


def test_track_ids_are_unique():
    t = IoUTracker()
    t.update(0, [_det(bbox=(0, 0, 10, 10)), _det(label="Buoy", bbox=(50, 50, 80, 80), z_order=50)])
    tracks = t.finalize()
    ids = [tr.track_id for tr in tracks]
    assert len(ids) == len(set(ids))


def test_track_id_starts_at_1():
    t = IoUTracker()
    t.update(0, [_det()])
    tracks = t.finalize()
    assert tracks[0].track_id == 1


# outside semantics

def test_track_lost_gets_outside_flag():
    t = IoUTracker()
    t.update(0, [_det(bbox=(10, 10, 50, 50))])  # frame 0 - present
    t.update(1, [])                             # frame 1 - missing -> outside
    tracks = t.finalize()
    assert len(tracks) == 1
    outside_kfs = [kf for kf in tracks[0].keyframes if kf.outside]
    assert len(outside_kfs) == 1
    assert outside_kfs[0].frame_idx == 1


def test_reappearing_object_starts_new_track():
    t = IoUTracker()
    t.update(0, [_det(bbox=(10, 10, 50, 50))])
    t.update(1, [])                               # lost - track 1 retired
    t.update(2, [_det(bbox=(12, 12, 52, 52))])    # re-enters - new track
    tracks = t.finalize()
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


#  cross-label isolation

def test_no_cross_label_matching():
    t = IoUTracker()
    t.update(0, [_det(label="Vessel", bbox=(10, 10, 50, 50), z_order=30)])
    # Same bbox, different label - must NOT match the Vessel track
    t.update(1, [_det(label="Buoy", bbox=(10, 10, 50, 50), z_order=50)])
    tracks = t.finalize()
    assert len(tracks) == 2
    labels = {tr.label for tr in tracks}
    assert labels == {"Vessel", "Buoy"}


# IoU threshold

def test_low_iou_starts_new_track():
    t = IoUTracker(iou_threshold=0.4)
    t.update(0, [_det(bbox=(0, 0, 10, 10))])
    # Completely non-overlapping bbox - new track
    t.update(1, [_det(bbox=(100, 100, 200, 200))])
    tracks = t.finalize()
    assert len(tracks) == 2


# bbox fallback polygon

def test_bbox_fallback_when_no_mask():
    t = IoUTracker()
    t.update(0, [_det(bbox=(5.0, 10.0, 25.0, 30.0), )])
    tracks = t.finalize()
    kf = tracks[0].keyframes[0]
    assert len(kf.points) == 4
    # Must form the bounding box rectangle
    xs = {p[0] for p in kf.points}
    ys = {p[1] for p in kf.points}
    assert xs == {5.0, 25.0}
    assert ys == {10.0, 30.0}


# finalize contracts

def test_finalize_empty_tracker():
    t = IoUTracker()
    assert t.finalize() == []

def test_finalize_returns_valid_track_annotations():
    from usv.auto_annotation.types import TrackAnnotation
    t = IoUTracker()
    t.update(0, [_det()])
    tracks = t.finalize()
    assert all(isinstance(tr, TrackAnnotation) for tr in tracks)
    assert all(len(tr.keyframes) > 0 for tr in tracks)
