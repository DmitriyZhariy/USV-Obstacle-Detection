import numpy as np

from usv.auto_annotation.tracker.iou_tracker import IoUTracker


_FRAME = np.zeros((64, 64, 3), dtype=np.uint8)


def _det(label="Vessel", bbox=(10, 10, 50, 50), z_order=30, conf=0.9):
    return {
        "label": label,
        "class_id": 5,
        "z_order": z_order,
        "confidence": conf,
        "bbox_xyxy": bbox,
        "mask": None,
    }


def _update(
    tracker: IoUTracker,
    frame_idx: int,
    detections: list[dict],
) -> None:
    """Call the current TrackerBase-compatible IoUTracker API."""
    tracker.update(frame_idx, _FRAME, detections)


# Basic lifecycle


def test_single_object_tracked_across_frames():
    tracker = IoUTracker()

    for frame_idx in range(5):
        _update(
            tracker,
            frame_idx,
            [_det(bbox=(10 + frame_idx, 10, 50 + frame_idx, 50))],
        )

    tracks = tracker.finalize()

    assert len(tracks) == 1
    assert tracks[0].label == "Vessel"
    assert len(tracks[0].keyframes) == 5
    assert all(keyframe.keyframe for keyframe in tracks[0].keyframes)


def test_track_ids_are_unique():
    tracker = IoUTracker()

    _update(
        tracker,
        0,
        [
            _det(bbox=(0, 0, 10, 10)),
            _det(label="Buoy", bbox=(50, 50, 80, 80), z_order=50),
        ],
    )

    tracks = tracker.finalize()
    track_ids = [track.track_id for track in tracks]

    assert len(track_ids) == len(set(track_ids))


def test_track_id_starts_at_1():
    tracker = IoUTracker()

    _update(tracker, 0, [_det()])
    tracks = tracker.finalize()

    assert tracks[0].track_id == 1


# Outside semantics


def test_track_lost_gets_outside_flag():
    tracker = IoUTracker(max_age=0)

    _update(tracker, 0, [_det(bbox=(10, 10, 50, 50))])
    _update(tracker, 1, [])

    tracks = tracker.finalize()
    outside_keyframes = [
        keyframe for keyframe in tracks[0].keyframes if keyframe.outside
    ]

    assert len(tracks) == 1
    assert len(outside_keyframes) == 1
    assert outside_keyframes[0].frame_idx == 1


def test_reappearing_object_starts_new_track():
    tracker = IoUTracker(max_age=0)

    _update(tracker, 0, [_det(bbox=(10, 10, 50, 50))])
    _update(tracker, 1, [])
    _update(tracker, 2, [_det(bbox=(12, 12, 52, 52))])

    tracks = tracker.finalize()

    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


def test_track_survives_until_max_age_is_exceeded():
    tracker = IoUTracker(max_age=2)

    _update(tracker, 0, [_det()])
    _update(tracker, 1, [])
    _update(tracker, 2, [])
    _update(tracker, 3, [])

    tracks = tracker.finalize()
    outside_keyframes = [
        keyframe for keyframe in tracks[0].keyframes if keyframe.outside
    ]

    assert len(tracks) == 1
    assert len(outside_keyframes) == 1
    assert outside_keyframes[0].frame_idx == 3


# Cross-label isolation


def test_no_cross_label_matching():
    tracker = IoUTracker()

    _update(
        tracker,
        0,
        [_det(label="Vessel", bbox=(10, 10, 50, 50), z_order=30)],
    )
    _update(
        tracker,
        1,
        [_det(label="Buoy", bbox=(10, 10, 50, 50), z_order=50)],
    )

    tracks = tracker.finalize()
    labels = {track.label for track in tracks}

    assert len(tracks) == 2
    assert labels == {"Vessel", "Buoy"}


# IoU threshold


def test_low_iou_starts_new_track():
    tracker = IoUTracker(iou_threshold=0.4)

    _update(tracker, 0, [_det(bbox=(0, 0, 10, 10))])
    _update(tracker, 1, [_det(bbox=(100, 100, 200, 200))])

    tracks = tracker.finalize()

    assert len(tracks) == 2


# Bounding-box fallback polygon


def test_bbox_fallback_when_no_mask():
    tracker = IoUTracker()

    _update(tracker, 0, [_det(bbox=(5.0, 10.0, 25.0, 30.0))])
    tracks = tracker.finalize()
    keyframe = tracks[0].keyframes[0]

    assert len(keyframe.points) == 4
    assert {point[0] for point in keyframe.points} == {5.0, 25.0}
    assert {point[1] for point in keyframe.points} == {10.0, 30.0}


# Finalize contracts


def test_finalize_empty_tracker():
    tracker = IoUTracker()

    assert tracker.finalize() == []


def test_finalize_returns_valid_track_annotations():
    from usv.auto_annotation.types import TrackAnnotation

    tracker = IoUTracker()

    _update(tracker, 0, [_det()])
    tracks = tracker.finalize()

    assert all(isinstance(track, TrackAnnotation) for track in tracks)
    assert all(track.keyframes for track in tracks)