"""Tests for pipeline mode routing — no model loading."""
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import numpy as np
import pandas as pd

from usv.auto_annotation.types import ClipData, TrackAnnotation, PolygonKeyframe


def _make_clip(n_frames: int = 5) -> ClipData:
    return ClipData(
        clip_name="mock_clip",
        frames=[np.zeros((64, 64, 3), dtype=np.uint8)] * n_frames,
        keyframe_idx=2,
        n_frames=n_frames,
        frame_height=64,
        frame_width=64,
        meta_df=pd.DataFrame(),
        original_video="mock.avi",
    )


def _make_track() -> TrackAnnotation:
    return TrackAnnotation(
        track_id=1, label="Vessel", z_order=30,
        keyframes=[PolygonKeyframe(
            frame_idx=2,
            points=[(0.,0.),(10.,0.),(10.,10.),(0.,10.)],
            keyframe=True, outside=False,
        )],
    )


@pytest.fixture
def pipeline_instance(tmp_path):
    """Создаём AutoAnnotationPipeline с полностью замоканными моделями."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("""
labels:
  - {name: Water,       group: stuff,  id: 0, z_order: 0}
  - {name: Sky,         group: stuff,  id: 1, z_order: 1}
  - {name: Vessel,      group: things, id: 5, z_order: 30}
  - {name: Buoy,        group: things, id: 8, z_order: 50}
min_instance_area: 64
""")
    # Патчим все тяжёлые зависимости
    with patch("usv.auto_annotation.pipeline.Florence2Detector"), \
         patch("usv.auto_annotation.pipeline.SAM2Segmentor"), \
         patch("usv.auto_annotation.pipeline.SAM2Tracker"), \
         patch("usv.auto_annotation.pipeline.SegFormerStuff"), \
         patch("pathlib.Path.exists", return_value=True):  # SAM2 checkpoint exists

        from usv.auto_annotation.pipeline import AutoAnnotationPipeline
        pipeline = AutoAnnotationPipeline(
            config_path=cfg_path,
            output_dir=tmp_path,
            mode="instance",
        )
    return pipeline


def test_instance_mode_skips_s5(pipeline_instance, tmp_path):
    """В instance mode SegFormerStuff не вызывается."""
    clip = _make_clip()
    track = _make_track()

    pipeline_instance._stage2_detect = MagicMock(return_value=[])
    pipeline_instance._stage3_segment = MagicMock(return_value=[])
    pipeline_instance._stage4_track = MagicMock(return_value=[track])
    pipeline_instance._stage5_stuff = MagicMock(return_value=[])
    pipeline_instance._stage8_export_coco = MagicMock(return_value=tmp_path / "out.json")
    pipeline_instance._write_manifest = MagicMock()

    pipeline_instance.run_clip(clip)

    pipeline_instance._stage5_stuff.assert_not_called()


def test_instance_mode_calls_coco_exporter(pipeline_instance, tmp_path):
    """В instance mode вызывается _stage8_export_coco, не _stage8_export."""
    clip = _make_clip()
    track = _make_track()

    pipeline_instance._stage2_detect = MagicMock(return_value=[])
    pipeline_instance._stage3_segment = MagicMock(return_value=[])
    pipeline_instance._stage4_track = MagicMock(return_value=[track])
    pipeline_instance._stage8_export = MagicMock()
    pipeline_instance._stage8_export_coco = MagicMock(return_value=tmp_path / "out.json")
    pipeline_instance._write_manifest = MagicMock()

    pipeline_instance.run_clip(clip)

    pipeline_instance._stage8_export_coco.assert_called_once()
    pipeline_instance._stage8_export.assert_not_called()