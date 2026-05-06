"""
YOLOv8-seg detector for the cpu-fast pipeline mode.
Maps COCO class names to project label names via the coco_mapping block
in auto_annotation.yaml. Classes absent from coco_mapping are silently dropped.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import yaml
import os

logger = logging.getLogger(__name__)


class YOLOv8Detector:
    """
    Wraps ultralytics YOLOv8n-seg for frame-level thing detection.

    Returns per-frame detection dicts for IoUTracker consumption:
        {
            "label":      str,                            # project label e.g. "Vessel"
            "class_id":   int,                            # project class ID from config
            "z_order":    int,                            # fixed from config
            "confidence": float,
            "bbox_xyxy":  tuple[float,float,float,float], # (x1,y1,x2,y2) original res
            "mask":       np.ndarray | None,              # H×W uint8 binary mask
        }
    """

    def __init__(
        self,
        config_path: Path,
        model_path: str = "yolov8n-seg.pt",
    ) -> None:
        from ultralytics import YOLO

        os.environ.setdefault("YOLO_AUTOINSTALL", "false")

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._coco_mapping: dict[str, str] = cfg.get("coco_mapping", {})
        if not self._coco_mapping:
            logger.warning(
                "YOLOv8Detector: coco_mapping is empty — no detections will be produced. "
                "Add coco_mapping block to %s", config_path
            )

        label_cfg: list[dict] = cfg["labels"]
        self._label_meta: dict[str, dict] = {
            lbl["name"]: {"id": lbl["id"], "z_order": lbl["z_order"]}
            for lbl in label_cfg
        }
        self._min_area: int = int(cfg.get("min_instance_area", 64))
        self._conf_threshold: float = float(cfg.get("detector_confidence", 0.25))

        self._model = YOLO(model_path)
        self._model.to("cpu")
        logger.info("YOLOv8Detector: loaded %s (CPU)", model_path)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run YOLOv8-seg on one BGR frame. Returns list of detection dicts."""
        results = self._model(frame, verbose=False, conf=self._conf_threshold)
        detections: list[dict] = []

        if not results or results[0].boxes is None:
            return detections

        result = results[0]
        h, w = frame.shape[:2]
        boxes = result.boxes
        masks = result.masks  # None when model produces no masks

        for i, box in enumerate(boxes):
            coco_name: str = result.names[int(box.cls)]
            project_label = self._coco_mapping.get(coco_name)
            if project_label is None:
                continue  # unmapped COCO class — skip silently

            meta = self._label_meta.get(project_label)
            if meta is None:
                logger.warning(
                    "coco_mapping target '%s' not found in config labels — skipping.",
                    project_label,
                )
                continue

            conf = float(box.conf)
            bbox = tuple(float(v) for v in box.xyxy[0].tolist())

            mask_arr: np.ndarray | None = None
            if masks is not None and i < len(masks.data):
                # Use masks.xy[i] — ultralytics returns contour points already
                # projected back to original image coordinates. Convert to binary mask.
                import cv2 as _cv2
                contour_pts = masks.xy[i] # shape (N, 2), already in original (w, h) space
                if len(contour_pts) >= 3:
                    bin_mask = np.zeros((h, w), dtype=np.uint8)
                    pts_int = contour_pts.astype(np.int32).reshape((-1, 1, 2))
                    _cv2.fillPoly(bin_mask, [pts_int], 1)
                    if int(np.count_nonzero(bin_mask)) >= self._min_area:
                        mask_arr = bin_mask

            detections.append({
                "label":      project_label,
                "class_id":   meta["id"],
                "z_order":    meta["z_order"],
                "confidence": conf,
                "bbox_xyxy":  bbox,
                "mask":       mask_arr,
            })

        return detections
