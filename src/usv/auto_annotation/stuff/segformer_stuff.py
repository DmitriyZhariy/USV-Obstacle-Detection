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
from collections import defaultdict

import numpy as np
import cv2

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

    def __init__(
            self, 
            model_name: str = _DEFAULT_MODEL,
            morph_kernel_size: int = 15,
            ) -> None:
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

        self._morph_kernel_size = morph_kernel_size
        if morph_kernel_size > 0:
            k = morph_kernel_size
            self._morph_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (k, k)
            )
        else:
            self._morph_kernel = None
        logger.info(
            "SegFormerStuff: morph_kernel_size=%d (%s)",
            morph_kernel_size,
            "ellipse" if morph_kernel_size > 0 else "disabled",
        )

    def _apply_morphology(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        CLOSE → OPEN на бинарной маске одного класса.
        CLOSE: закрывает дырки внутри региона.
        OPEN:  убирает шумовые острова вне региона.
        Работает in-place через временный буфер.
        """
        if self._morph_kernel is None:
            return binary_mask
        closed = cv2.morphologyEx(
            binary_mask, cv2.MORPH_CLOSE, self._morph_kernel
        )
        opened = cv2.morphologyEx(
            closed, cv2.MORPH_OPEN, self._morph_kernel
        )
        return opened
    
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

        class_id_to_mask: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros((h, w), dtype=np.uint8)
        )

        for ade_id in np.unique(ade20k_map):
            result = get_project_label(int(ade_id))
            if result is None:
                continue
            _label, class_id, _z_order = result
            class_id_to_mask[class_id] |= (ade20k_map == ade_id).astype(np.uint8)

        for class_id, binary_mask in class_id_to_mask.items():
            clean_mask = self._apply_morphology(binary_mask)
            project_map[clean_mask > 0] = class_id

        unique, counts = np.unique(ade20k_map, return_counts=True)
        total = h * w
        logger.info("  SegFormer ADE20K distribution (top-10):")
        for ade_id, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
            result = get_project_label(int(ade_id))
            label = result[0] if result else f"ADE#{ade_id}"
            logger.info("    ade_id=%3d  %-20s  %5.1f%%", ade_id, label, 100*cnt/total)

        return project_map
