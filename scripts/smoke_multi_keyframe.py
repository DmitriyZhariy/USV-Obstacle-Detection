"""
Smoke-test: run _run_segmented_tracking with a mocked SAM2Tracker.
No models, no GPU. Just verifies the segment loop and stitching logic.
"""
from unittest.mock import MagicMock, patch
import numpy as np
from usv.auto_annotation.types import TrackAnnotation, PolygonKeyframe, ClipData

# Синтетические кадры: 120 штук, 64x64 px
N_FRAMES = 120
frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(N_FRAMES)]

clip_data = ClipData(
    clip_name="smoke_test",
    frames=frames,
    keyframe_idx=N_FRAMES // 2,
    n_frames=N_FRAMES,
    frame_height=64,
    frame_width=64,
    meta_df=None,
    original_video="",
)

# Mock: детектор всегда возвращает одну детекцию
mock_detection = [{"label": "Vessel", "class_id": 5, "z_order": 30,
                   "confidence": 0.9, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]}]
mock_mask = [{"track_id": 1, "label": "Vessel", "class_id": 5, "z_order": 30,
              "confidence": 0.9, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
              "mask": np.ones((64, 64), dtype=np.uint8)}]

# Mock: трекер возвращает один трек с одним keyframe
def fake_finalize():
    seg_start = fake_tracker._segment_start
    return [TrackAnnotation(
        track_id=1, label="Vessel", z_order=30,
        keyframes=[PolygonKeyframe(
            frame_idx=seg_start + 12,  # середина сегмента 25 // 2
            points=[(10,10),(50,10),(50,50),(10,50)],
            keyframe=True,
        )]
    )]

fake_tracker = MagicMock()
fake_tracker._segment_start = 0
fake_tracker.init_clip.side_effect = lambda frames, keyframe_idx, instance_masks, segment_start=0: setattr(fake_tracker, "_segment_start", segment_start)
fake_tracker.finalize.side_effect = fake_finalize

with patch("usv.auto_annotation.pipeline.AutoAnnotationPipeline.__init__", return_value=None):
    from usv.auto_annotation.pipeline import AutoAnnotationPipeline
    p = AutoAnnotationPipeline.__new__(AutoAnnotationPipeline)
    p._tracker = fake_tracker
    p._detector = MagicMock(return_value=mock_detection)
    p._segmentor = MagicMock()
    p._cfg = {"multi_keyframe_interval": 50, "min_instance_area_frac": 0.0,
              "min_instance_area": 64}

    p._stage2_detect = MagicMock(return_value=mock_detection)
    p._stage3_segment = MagicMock(return_value=mock_mask)

    tracks = p._run_segmented_tracking(clip_data, interval=50)

print(f"Tracks produced: {len(tracks)}")
for t in tracks:
    kf_idxs = [k.frame_idx for k in t.keyframes]
    print(f"  track_id={t.track_id} label={t.label} keyframes={kf_idxs}")

# Ожидаем: 1 трек, keyframes из разных сегментов (0, 50, 100)
# с абсолютными индексами (12, 62, 112)
assert len(tracks) == 1, f"Expected 1 track, got {len(tracks)}"
kf_idxs = sorted(k.frame_idx for k in tracks[0].keyframes)
assert kf_idxs == sorted(set(kf_idxs)), "Duplicate frame_idx in keyframes"
print("\n✓ Smoke-test passed")