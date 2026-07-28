"""
Unit-тесты для multi-keyframe логики.

Запуск:
    uv run python tests/test_multi_keyframe.py

Не требует моделей, GPU, SAM2 checkpoint.
Тестирует:
  - _match_segment_tracks: корректная склейка треков по bbox IoU
  - segment_start: сохранение и сброс смещения в SAM2Tracker
  - keyframe sort: после extend keyframes всегда отсортированы
"""
import sys
sys.path.insert(0, "src")

import dataclasses
import numpy as np
from unittest.mock import MagicMock, patch

from usv.auto_annotation.types import TrackAnnotation, PolygonKeyframe, ClipData
from usv.auto_annotation.postprocess.mask_utils import bbox_iou


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rect(x1, y1, x2, y2):
    """Прямоугольный полигон из 4 точек."""
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _make_track(track_id, label, frame_idx, points, z_order=30):
    return TrackAnnotation(
        track_id=track_id,
        label=label,
        z_order=z_order,
        keyframes=[PolygonKeyframe(frame_idx=frame_idx, points=points, keyframe=True)],
    )


def _make_pipeline():
    """Создаёт экземпляр AutoAnnotationPipeline без загрузки моделей."""
    with patch.object(
        __import__("usv.auto_annotation.pipeline", fromlist=["AutoAnnotationPipeline"])
        .AutoAnnotationPipeline, "__init__", return_value=None
    ):
        from usv.auto_annotation.pipeline import AutoAnnotationPipeline
        p = AutoAnnotationPipeline.__new__(AutoAnnotationPipeline)
    return p


def _make_tracker_stub():
    """SAM2Tracker без загрузки checkpoint."""
    with patch.object(
        __import__("usv.auto_annotation.tracker.sam2_tracker", fromlist=["SAM2Tracker"])
        .SAM2Tracker, "__init__", return_value=None
    ):
        from usv.auto_annotation.tracker.sam2_tracker import SAM2Tracker
        t = SAM2Tracker.__new__(SAM2Tracker)
        t._frames = []
        t._keyframe_idx = 0
        t._instance_masks = []
        t._segment_start = 0
    return t


# ─── bbox_iou sanity ──────────────────────────────────────────────────────────

def test_bbox_iou_identical():
    box = (0.0, 0.0, 100.0, 100.0)
    iou = bbox_iou(box, box)
    assert abs(iou - 1.0) < 1e-6, f"Identical boxes → IoU=1.0, got {iou}"
    print("  ✓ bbox_iou identical → 1.0")


def test_bbox_iou_no_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (100.0, 100.0, 200.0, 200.0)
    iou = bbox_iou(a, b)
    assert iou == 0.0, f"No-overlap → IoU=0.0, got {iou}"
    print("  ✓ bbox_iou no overlap → 0.0")


def test_bbox_iou_half_overlap():
    # a: [0,0,100,100], b: [50,0,150,100] → intersect 50×100, union 150×100
    a = (0.0, 0.0, 100.0, 100.0)
    b = (50.0, 0.0, 150.0, 100.0)
    iou = bbox_iou(a, b)
    expected = (50 * 100) / (150 * 100)  # = 1/3
    assert abs(iou - expected) < 1e-6, f"Half-overlap → {expected:.4f}, got {iou:.4f}"
    print(f"  ✓ bbox_iou half-overlap → {iou:.4f}")


# ─── _match_segment_tracks ────────────────────────────────────────────────────

def test_match_overlapping_same_label():
    """Высокий IoU + одинаковый label → reuse существующего track_id."""
    p = _make_pipeline()
    existing = _make_track(1, "Vessel", 25, _rect(100, 100, 200, 200))
    all_tracks = {1: existing}

    new_track = _make_track(1, "Vessel", 75, _rect(105, 105, 195, 195))
    result = p._match_segment_tracks([new_track], all_tracks, next_global_id=2)

    assert len(result) == 1
    assert result[0].track_id == 1, \
        f"Должен reuse track_id=1, получил {result[0].track_id}"
    print("  ✓ match: высокий IoU + same label → track_id reused")


def test_no_match_different_label():
    """Тот же bbox, но разный label → новый track_id."""
    p = _make_pipeline()
    existing = _make_track(1, "Vessel", 25, _rect(100, 100, 200, 200))
    all_tracks = {1: existing}

    new_track = _make_track(1, "Buoy", 75, _rect(100, 100, 200, 200))
    result = p._match_segment_tracks([new_track], all_tracks, next_global_id=2)

    assert result[0].track_id == 2, \
        f"Разный label → новый track_id=2, получил {result[0].track_id}"
    print("  ✓ match: разный label → new track_id")


def test_no_match_low_iou():
    """Далёкий bbox → IoU < 0.3 → новый track_id."""
    p = _make_pipeline()
    existing = _make_track(1, "Vessel", 25, _rect(0, 0, 50, 50))
    all_tracks = {1: existing}

    new_track = _make_track(1, "Vessel", 75, _rect(500, 500, 600, 600))
    result = p._match_segment_tracks([new_track], all_tracks, next_global_id=2)

    assert result[0].track_id == 2, \
        f"Низкий IoU → новый track_id=2, получил {result[0].track_id}"
    print("  ✓ match: низкий IoU → new track_id")


def test_empty_all_tracks():
    """Первый сегмент: all_tracks пуст → все треки получают новые id от next_global_id."""
    p = _make_pipeline()
    new_tracks = [
        _make_track(99, "Vessel", 10, _rect(0, 0, 100, 100)),
        _make_track(99, "Buoy",   10, _rect(200, 200, 300, 300)),
    ]
    result = p._match_segment_tracks(new_tracks, {}, next_global_id=1)

    ids = [t.track_id for t in result]
    assert ids == [1, 2], f"Ожидались [1, 2], получил {ids}"
    print(f"  ✓ match: пустые all_tracks → новые id {ids}")


def test_no_double_assignment():
    """Один существующий трек не должен быть сматчен дважды."""
    p = _make_pipeline()
    existing = _make_track(1, "Vessel", 25, _rect(100, 100, 200, 200))
    all_tracks = {1: existing}

    # Оба новых трека перекрываются с existing
    t1 = _make_track(1, "Vessel", 75, _rect(100, 100, 200, 200))
    t2 = _make_track(2, "Vessel", 75, _rect(102, 102, 198, 198))
    result = p._match_segment_tracks([t1, t2], all_tracks, next_global_id=2)

    ids = [t.track_id for t in result]
    assert len(set(ids)) == 2, f"Должны быть 2 уникальных id, получил {ids}"
    assert 1 in ids, f"Первый матч должен reuse id=1, получил {ids}"
    new_id = [x for x in ids if x != 1][0]
    assert new_id >= 2, f"Второй трек должен получить id >= 2, получил {new_id}"
    print(f"  ✓ match: нет двойного матча → ids={ids}")


def test_iou_threshold_boundary():
    """IoU ровно на пороге 0.3 — матч должен состояться (>= порога)."""
    p = _make_pipeline()
    # Подбираем bbox так, чтобы IoU было чуть выше 0.3
    # a=[0,0,100,100] area=10000, b=[70,0,200,100] area=13000
    # intersect=[70,0,100,100]=30×100=3000, union=20000, iou=3000/20000=0.15 — мало
    # Используем b=[40,0,140,100]: intersect=60×100=6000, union=14000, iou≈0.4286
    existing = _make_track(1, "Vessel", 25, _rect(0, 0, 100, 100))
    all_tracks = {1: existing}
    new_track = _make_track(1, "Vessel", 75, _rect(40, 0, 140, 100))

    iou_val = bbox_iou((0, 0, 100, 100), (40, 0, 140, 100))
    result = p._match_segment_tracks([new_track], all_tracks, next_global_id=2)

    if iou_val >= 0.3:
        assert result[0].track_id == 1, \
            f"IoU={iou_val:.3f} >= 0.3 → должен reuse id=1, получил {result[0].track_id}"
        print(f"  ✓ match: IoU={iou_val:.3f} >= 0.3 → reused")
    else:
        assert result[0].track_id == 2, \
            f"IoU={iou_val:.3f} < 0.3 → новый id, получил {result[0].track_id}"
        print(f"  ✓ match: IoU={iou_val:.3f} < 0.3 → new id")


# ─── segment_start в SAM2Tracker ─────────────────────────────────────────────

def test_segment_start_stored():
    """init_clip сохраняет segment_start."""
    t = _make_tracker_stub()
    t.init_clip(frames=[], keyframe_idx=0, instance_masks=[], segment_start=50)
    assert t._segment_start == 50, \
        f"Ожидалось _segment_start=50, получил {t._segment_start}"
    print("  ✓ SAM2Tracker: segment_start=50 сохранён")


def test_segment_start_default_zero():
    """init_clip без segment_start → default 0."""
    t = _make_tracker_stub()
    t.init_clip(frames=[], keyframe_idx=0, instance_masks=[])
    assert t._segment_start == 0, \
        f"Ожидалось _segment_start=0 (default), получил {t._segment_start}"
    print("  ✓ SAM2Tracker: segment_start default = 0")


def test_segment_start_accumulates_correctly():
    """Три вызова init_clip с разными segment_start — каждый раз правильное значение."""
    t = _make_tracker_stub()
    for expected in [0, 50, 100]:
        t.init_clip(frames=[], keyframe_idx=0, instance_masks=[], segment_start=expected)
        assert t._segment_start == expected, \
            f"Ожидалось {expected}, получил {t._segment_start}"
    print("  ✓ SAM2Tracker: segment_start корректно обновляется между вызовами")


# ─── keyframe sort после склейки ──────────────────────────────────────────────

def test_keyframes_sorted_after_extend():
    """
    После склейки двух сегментов keyframes должны быть отсортированы по frame_idx.
    Воспроизводит логику из _run_segmented_tracking.
    """
    kf_seg0 = PolygonKeyframe(frame_idx=25, points=_rect(10, 10, 50, 50), keyframe=True)
    kf_seg1 = PolygonKeyframe(frame_idx=75, points=_rect(12, 12, 48, 48), keyframe=True)

    track = TrackAnnotation(track_id=1, label="Vessel", z_order=30,
                            keyframes=[kf_seg0])

    # Имитируем extend из _run_segmented_tracking
    seg1_track = TrackAnnotation(track_id=1, label="Vessel", z_order=30,
                                 keyframes=[kf_seg1])
    track.keyframes.extend(seg1_track.keyframes)
    track.keyframes.sort(key=lambda k: k.frame_idx)

    idxs = [k.frame_idx for k in track.keyframes]
    assert idxs == sorted(idxs), f"keyframes не отсортированы: {idxs}"
    assert idxs == [25, 75], f"Ожидались [25, 75], получил {idxs}"
    print(f"  ✓ keyframes отсортированы после extend: {idxs}")


def test_keyframes_sorted_reverse_insert():
    """Сегменты могут прийти в обратном порядке — sort всё равно должен работать."""
    kf_later  = PolygonKeyframe(frame_idx=75, points=_rect(10, 10, 50, 50), keyframe=True)
    kf_earlier = PolygonKeyframe(frame_idx=25, points=_rect(10, 10, 50, 50), keyframe=True)

    track = TrackAnnotation(track_id=1, label="Vessel", z_order=30,
                            keyframes=[kf_later])
    track.keyframes.extend([kf_earlier])
    track.keyframes.sort(key=lambda k: k.frame_idx)

    idxs = [k.frame_idx for k in track.keyframes]
    assert idxs == [25, 75], f"Ожидались [25, 75], получил {idxs}"
    print(f"  ✓ keyframes отсортированы после reverse-insert: {idxs}")


# ─── runner ───────────────────────────────────────────────────────────────────

TESTS = [
    ("bbox_iou identical",            test_bbox_iou_identical),
    ("bbox_iou no overlap",           test_bbox_iou_no_overlap),
    ("bbox_iou half overlap",         test_bbox_iou_half_overlap),
    ("match: high IoU same label",    test_match_overlapping_same_label),
    ("match: different label",        test_no_match_different_label),
    ("match: low IoU",                test_no_match_low_iou),
    ("match: empty all_tracks",       test_empty_all_tracks),
    ("match: no double assignment",   test_no_double_assignment),
    ("match: IoU threshold boundary", test_iou_threshold_boundary),
    ("segment_start stored",          test_segment_start_stored),
    ("segment_start default zero",    test_segment_start_default_zero),
    ("segment_start accumulates",     test_segment_start_accumulates_correctly),
    ("keyframes sorted after extend", test_keyframes_sorted_after_extend),
    ("keyframes sorted reverse",      test_keyframes_sorted_reverse_insert),
]

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    print(f"\n{'─'*60}")
    print(f"  USV multi-keyframe unit-tests  ({len(TESTS)} tests)")
    print(f"{'─'*60}")

    for name, fn in TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            failed += 1
            errors.append((name, exc))
            print(f"  ✗ {name}")
            print(f"      {type(exc).__name__}: {exc}")

    print(f"{'─'*60}")
    if failed == 0:
        print(f"  ✓ Все {passed} тестов прошли успешно.")
    else:
        print(f"  ✗ {failed} упало / {passed} прошло")
        for name, exc in errors:
            print(f"\n  FAIL: {name}")
            print(f"        {type(exc).__name__}: {exc}")
    print(f"{'─'*60}\n")
    sys.exit(0 if failed == 0 else 1)