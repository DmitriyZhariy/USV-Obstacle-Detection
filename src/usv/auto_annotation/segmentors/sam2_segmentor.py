"""
SAM 2.1 image segmentor for the cpu-sam2 pipeline mode.

Model : sam2.1_hiera_small (image predictor, NOT video predictor)
Task  : box-prompted instance segmentation on a single keyframe

Used in Stage 3 of the cpu-sam2 pipeline:
    Florence2Detector  →  bboxes  →  SAM2Segmentor  →  binary masks
    SAM2Tracker then propagates those keyframe masks across all frames.

Output contract
---------------
Returns one H×W uint8 binary mask (values 0/1) per input bounding box,
in the same order as bboxes_xyxy. If SAM 2 produces an empty or
sub-threshold mask for a box, that slot is a zero-filled array.
Callers (pipeline.py) discard slots where mask_area() < min_instance_area.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from usv.auto_annotation.segmentors.base import InstanceSegmentorBase

logger = logging.getLogger(__name__)


class SAM2Segmentor(InstanceSegmentorBase):
    """
    Wraps SAM2ImagePredictor (hiera_small) for box-prompted segmentation.

    One predictor instance is created at construction and reused across
    all keyframes in a clip. The predictor holds no frame state between
    calls to segment() — each call is fully independent.

    CPU inference only. torch.inference_mode() is applied per-call.
    """

    def __init__(
        self,
        checkpoint: Path | str,
        model_cfg: str = "sam2.1_hiera_s",
    ) -> None:
        """
        Parameters
        ----------
        checkpoint : Path | str
            Path to the .pt checkpoint file, e.g. models/sam2.1_hiera_small.pt
        model_cfg : str
            Config name string passed to build_sam2().
            Must match the checkpoint variant. Do not pass a filesystem path —
            the sam2 package resolves bundled YAML configs by name internally.
        """
        import os
        import torch
        import sam2
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        logger.info(
            "SAM2Segmentor: loading %s from %s (CPU) ...", model_cfg, checkpoint
        )

        sam2_cfg_dir = os.path.join(os.path.dirname(sam2.__file__), "configs", "sam2.1")
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=sam2_cfg_dir, job_name="sam2", version_base=None)

        sam2_model = build_sam2(
            config_file=model_cfg,
            ckpt_path=str(checkpoint),
            device="cpu",
            apply_postprocessing=False,
        )
        GlobalHydra.instance().clear()

        self._predictor = SAM2ImagePredictor(sam2_model)
        self._torch = torch
        logger.info("SAM2Segmentor: ready.")

    def segment(
        self,
        frame: np.ndarray,
        bboxes_xyxy: list[list[int]],
    ) -> list[np.ndarray]:
        """
        Box-prompted segmentation on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 (H, W, 3) at original clip resolution.
        bboxes_xyxy : list[list[int]]
            List of [x1, y1, x2, y2] boxes in original pixel coordinates.

        Returns
        -------
        list[np.ndarray]
            One H×W uint8 binary mask per input box, same order.
            Returns empty list if bboxes_xyxy is empty.
        """
        if not bboxes_xyxy:
            return []

        import cv2

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes_np = np.array(bboxes_xyxy, dtype=np.float32)  # (N, 4)

        with self._torch.inference_mode():
            self._predictor.set_image(rgb)
            masks_out, scores, _ = self._predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_np,
                multimask_output=False,  # one mask per box
            )

        # masks_out shape: (N, 1, H, W) bool when multimask_output=False
        # For a single box SAM2 returns (1, H, W); for multiple (N, 1, H, W).
        # Normalise to always (N, H, W) uint8.
        if masks_out.ndim == 3:
            # Single box case: (1, H, W) → add batch dim → (1, 1, H, W)
            masks_out = masks_out[np.newaxis, ...]

        results: list[np.ndarray] = []
        for i in range(len(bboxes_xyxy)):
            if i < masks_out.shape[0]:
                # shape (1, H, W) → squeeze → (H, W), convert bool→uint8
                mask = masks_out[i, 0].astype(np.uint8)
            else:
                # SAM 2 returned fewer masks than boxes — fill with zeros
                logger.warning(
                    "SAM2Segmentor: SAM2 returned %d masks for %d boxes "
                    "— filling slot %d with zeros",
                    masks_out.shape[0], len(bboxes_xyxy), i,
                )
                mask = np.zeros((h, w), dtype=np.uint8)

            results.append(mask)

        return results
