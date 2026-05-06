"""
SegFormer-B0 stuff segmentor for the cpu-sam2 pipeline mode.

Model  : nvidia/segformer-b0-finetuned-ade-512-512
Dataset: ADE20K (150 classes)

Output pixel map contract
--------------------------
Returns H×W uint8 ndarray where each pixel value is a project class_id
(from configs/auto_annotation.yaml labels[].id), NOT a raw ADE20K ID
and NOT a z_order value.

    0   → Water
    1   → Sky
    2   → Land
    4   → Bridge
    255 → unmapped / no stuff class (sentinel)

overlap_resolver.py reads class_id from this map and resolves label +
z_order via the config label table. class_id is unique per label, so
the map is unambiguous even when two labels share a z_order (Pier/Bridge).
"""
from __future__ import annotations

import logging

import numpy as np

from usv.auto_annotation.stuff.base import StuffSegmentorBase
from usv.auto_annotation.stuff.ade20k_mapping import get_project_label

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"


class SegFormerStuff(StuffSegmentorBase):
    """
    Wraps HuggingFace SegformerForSemanticSegmentation (B0, ADE20K).

    Maps ADE20K class IDs → project class_id values via ade20k_mapping.py.
    Unmapped pixels are set to 255.

    CPU inference only. Model loaded once at construction, reused per frame.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        from transformers import (
            SegformerImageProcessor,
            SegformerForSemanticSegmentation,
        )
        import torch

        logger.info("SegFormerStuff: loading %s on CPU ...", model_name)
        self._processor = SegformerImageProcessor.from_pretrained(model_name)
        self._model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self._model.eval()
        self._model.to("cpu")
        self._torch = torch
        logger.info("SegFormerStuff: ready.")

    def segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment one BGR frame into a project class_id map.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 (H, W, 3) at original clip resolution.

        Returns
        -------
        np.ndarray
            uint8 (H, W) — project class_id per pixel; 255 = unmapped.
        """
        import cv2

        h, w = frame.shape[:2]

        # SegFormer processor expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        # logits: (1, 150, H/4, W/4) → upsample to original resolution
        logits = outputs.logits
        upsampled = self._torch.nn.functional.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        # ADE20K class ID per pixel, numpy int32, shape (H, W)
        ade20k_map: np.ndarray = (
            upsampled.argmax(dim=1).squeeze(0).numpy().astype(np.int32)
        )

        # Build output filled with 255 (unmapped sentinel)
        project_map = np.full((h, w), 255, dtype=np.uint8)

        for ade_id in np.unique(ade20k_map):
            result = get_project_label(int(ade_id))
            if result is None:
                continue  # ADE20K class not in USV ontology — leave as 255

            _label, class_id, _z_order = result
            # Store class_id — unique per label, unambiguous even for
            # Pier/Bridge which share z_order=10
            project_map[ade20k_map == ade_id] = class_id

        return project_map
