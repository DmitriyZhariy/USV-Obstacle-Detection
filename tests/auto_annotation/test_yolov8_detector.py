# tests/auto_annotation/test_yolov8_detector.py

import numpy as np
import pytest
from pathlib import Path
from usv.auto_annotation.detectors.yolov8_detector import YOLOv8Detector

CONFIG = Path("configs/auto_annotation.yaml")
MODEL  = Path("models/yolov8s-seg.pt")

pytestmark = pytest.mark.skipif(
    not MODEL.exists(),
    reason=f"{MODEL} not found in models/ - download manually first",
)


@pytest.fixture(scope="module")
def detector():
    return YOLOv8Detector(config_path=CONFIG, model_path=str(MODEL))


def test_detector_loads(detector):
    assert detector is not None


def test_detect_returns_list_on_blank_frame(detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = detector.detect(frame)
    assert isinstance(dets, list)


def test_detect_dict_keys_present(detector):
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    dets = detector.detect(frame)
    for det in dets:
        assert "label" in det
        assert "class_id" in det
        assert "z_order" in det
        assert "confidence" in det
        assert "bbox_xyxy" in det
        assert "mask" in det
        assert det["label"] in {"Vessel", "Other", "Buoy", "LandingMark", "BridgeLight"}


def test_detect_on_real_frame_if_available(detector):
    clips_dir = Path("data/interim/choosed_clips_v5-1/frames")
    if not clips_dir.exists():
        pytest.skip("No real clips available")

    clips = sorted(clips_dir.iterdir())
    if not clips:
        pytest.skip("frames/ directory is empty")

    frame_files = sorted(clips[0].glob("*.jpeg"))
    if not frame_files:
        pytest.skip("No JPEG frames found")

    import cv2
    frame = cv2.imread(str(frame_files[0]))
    assert frame is not None

    dets = detector.detect(frame)
    assert isinstance(dets, list)
    for det in dets:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        assert x2 > x1 and y2 > y1
        assert 0.0 <= det["confidence"] <= 1.0
        if det["mask"] is not None:
            assert det["mask"].shape == (frame.shape[0], frame.shape[1])
            assert det["mask"].dtype == np.uint8
