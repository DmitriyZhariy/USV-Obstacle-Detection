from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from usv.auto_annotation.types import ClipData, PolygonKeyframe, TrackAnnotation


def test_segmented_tracking_stitches_absolute_keyframes() -> None:
    from usv.auto_annotation.pipeline import AutoAnnotationPipeline

    n_frames = 120
    frames = [
        np.zeros((64, 64, 3), dtype=np.uint8)
        for _ in range(n_frames)
    ]
    clip_data = ClipData(
        clip_name="smoke_test",
        frames=frames,
        keyframe_idx=n_frames // 2,
        n_frames=n_frames,
        frame_height=64,
        frame_width=64,
        meta_df=None,
        original_video="",
    )

    mock_detection = [
        {
            "label": "Vessel",
            "class_id": 5,
            "z_order": 30,
            "confidence": 0.9,
            "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
        }
    ]
    mock_mask = [
        {
            "track_id": 1,
            "label": "Vessel",
            "class_id": 5,
            "z_order": 30,
            "confidence": 0.9,
            "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
            "mask": np.ones((64, 64), dtype=np.uint8),
        }
    ]

    tracker = MagicMock()
    tracker._segment_start = 0

    def init_clip(
        frames,
        keyframe_idx,
        instance_masks,
        segment_start=0,
    ) -> None:
        tracker._segment_start = segment_start

    def finalize() -> list[TrackAnnotation]:
        segment_start = tracker._segment_start
        return [
            TrackAnnotation(
                track_id=1,
                label="Vessel",
                z_order=30,
                keyframes=[
                    PolygonKeyframe(
                        frame_idx=segment_start + 12,
                        points=[(10, 10), (50, 10), (50, 50), (10, 50)],
                        keyframe=True,
                    )
                ],
            )
        ]

    tracker.init_clip.side_effect = init_clip
    tracker.finalize.side_effect = finalize

    with patch.object(
        AutoAnnotationPipeline,
        "__init__",
        return_value=None,
    ):
        pipeline = AutoAnnotationPipeline.__new__(AutoAnnotationPipeline)

    pipeline._tracker = tracker
    pipeline._detector = MagicMock(return_value=mock_detection)
    pipeline._segmentor = MagicMock()
    pipeline._cfg = {
        "multi_keyframe_interval": 50,
        "min_instance_area_frac": 0.0,
        "min_instance_area": 64,
    }
    pipeline._stage2_detect = MagicMock(return_value=mock_detection)
    pipeline._stage3_segment = MagicMock(return_value=mock_mask)

    tracks = pipeline._run_segmented_tracking(clip_data, interval=50)

    assert tracker.init_clip.call_count == 3
    assert [
        call.kwargs["segment_start"]
        for call in tracker.init_clip.call_args_list
    ] == [0, 50, 100]

    assert len(tracks) == 1
    assert tracks[0].label == "Vessel"
    assert tracks[0].track_id == 1

    keyframe_indices = [keyframe.frame_idx for keyframe in tracks[0].keyframes]
    assert keyframe_indices == [12, 62, 112]
    assert len(keyframe_indices) == len(set(keyframe_indices))