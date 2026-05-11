"""
Florence-2-base open-vocabulary detector for the cpu-sam2 pipeline mode.

Model : microsoft/Florence-2-base (HuggingFace transformers)
Task  : <OD> open-vocabulary detection using project thing-class label names
        as the prompt.

Bbox rescale contract
---------------------
Florence-2 internally normalises all coordinates to a 1000×1000 grid
regardless of the actual input image size. The rescale formula is:

    x_orig = x_1000 / 1000.0 * orig_W
    y_orig = y_1000 / 1000.0 * orig_H

Do NOT divide by inference_resize — that is a common off-by-one error.
The frame IS resized before inference (long-side → inference_resize) to
reduce latency, but Florence-2 still reports coordinates in 0–1000 space,
so the only correct denominator is always 1000.

Output dict schema (identical to YOLOv8Detector)
-------------------------------------------------
    label      : str
    class_id   : int
    z_order    : int
    confidence : float   — Florence-2 has no per-box score; fixed at 1.0
    bbox_xyxy  : tuple[float, float, float, float]  original resolution
    mask       : None    — masks come from SAM2Segmentor in Stage 3
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from usv.auto_annotation.detectors.base import DetectorBase

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "microsoft/Florence-2-base"
_FLORENCE_GRID = 1000.0   # Florence-2 normalises to this grid — never change


class Florence2Detector(DetectorBase):
    """
    Wraps Florence-2-base for open-vocabulary thing detection.

    Prompts the model with all configured thing-class label names joined
    as a comma-separated string. Returns one detection dict per predicted
    bounding box, rescaled to original frame resolution.

    Notes
    -----
    - Florence-2 does not output per-box confidence scores.
      All returned dicts have confidence=1.0.
    - mask is always None — SAM2Segmentor handles mask generation.
    - CPU inference only; no CUDA code paths.
    """

    def __init__(
        self,
        config_path: Path,
        model_name: str = _DEFAULT_MODEL,
        inference_resize: int = 640,
    ) -> None:
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Build label metadata for thing classes only
        self._label_meta: dict[str, dict] = {
            lbl["name"]: {"id": lbl["id"], "z_order": lbl["z_order"]}
            for lbl in cfg["labels"]
            if lbl["group"] == "things"
        }
        if not self._label_meta:
            raise ValueError(
                f"Florence2Detector: no thing-class labels found in {config_path}"
            )

        # Build the prompt once — reused for every frame
        # Format: "Vessel . Buoy . LandingMark . BridgeLight . Other"
        self._prompt = " . ".join(self._label_meta.keys())
        self._task_token = "<OD>"
        self._inference_resize = inference_resize

        logger.info(
            "Florence2Detector: loading %s (CPU) ...", model_name
        )
        self._processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
        ).float() # force fp32 — "auto" picks fp16 on CPU, which breaks conv ops
        self._model.eval()
        self._model.to("cpu")
        self._torch = torch
        logger.info(
            "Florence2Detector: ready. Thing classes: %s",
            list(self._label_meta.keys()),
        )

    # Internal helpers

    def _resize_for_inference(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Resize frame so the long side equals self._inference_resize.

        Returns
        -------
        resized : np.ndarray   — RGB uint8
        scale   : float        — NOT used for bbox rescaling (see module docstring);
                                 kept for debug / logging only
        """
        h, w = frame.shape[:2]
        long_side = max(h, w)
        scale = self._inference_resize / long_side
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale

    def _rescale_bbox(
        self,
        bbox_1000: list[float],
        orig_h: int,
        orig_w: int,
    ) -> tuple[float, float, float, float]:
        """
        Rescale a Florence-2 bbox from 0–1000 grid to original pixel coords.

        Florence-2 always reports in 0–1000 regardless of input image size.
        Formula: x_orig = x_1000 / 1000 * orig_W  (same for y with orig_H).
        """
        x1, y1, x2, y2 = bbox_1000
        sx = orig_w / _FLORENCE_GRID
        sy = orig_h / _FLORENCE_GRID
        x1_orig = max(0.0, x1 * sx)
        y1_orig = max(0.0, y1 * sy)
        x2_orig = min(float(orig_w), x2 * sx)
        y2_orig = min(float(orig_h), y2 * sy)
        return (x1_orig, y1_orig, x2_orig, y2_orig)

    # DetectorBase interface

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run Florence-2 open-vocabulary detection on one BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 (H, W, 3) at original clip resolution.

        Returns
        -------
        list[dict]
            Detection dicts matching DetectorBase schema.
            Empty list if Florence-2 finds nothing or output is malformed.
        """
        orig_h, orig_w = frame.shape[:2]
        resized_rgb, _scale = self._resize_for_inference(frame)

        from PIL import Image as _PIL_Image
        pil_img = _PIL_Image.fromarray(resized_rgb)

        inputs = self._processor(
            text=f"<CAPTION_TO_PHRASE_GROUNDING> {self._prompt}",
            images=pil_img,
            return_tensors="pt",
        )
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                use_cache=False,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Post-process: parse Florence-2 structured output
        parsed = self._processor.post_process_generation(
            generated_text,
            task="<CAPTION_TO_PHRASE_GROUNDING>",
            image_size=(pil_img.width, pil_img.height),
        )

        # parsed[<CAPTION_TO_PHRASE_GROUNDING>] = {"bboxes": [[x1,y1,x2,y2], ...], "labels": ["...", ...]}
        od_result = parsed.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
        raw_bboxes: list[list[float]] = od_result.get("bboxes", [])
        raw_labels: list[str] = od_result.get("labels", [])

        if not raw_bboxes:
            logger.debug(
                "Florence2Detector: no detections on frame (%dx%d)",
                orig_w, orig_h,
            )
            return []

        detections: list[dict] = []
        for bbox_1000, raw_label in zip(raw_bboxes, raw_labels):
            # Match Florence-2 label to nearest project thing label
            # Florence-2 may return the exact prompt word or a close variant
            project_label = self._match_label(raw_label)
            if project_label is None:
                logger.debug(
                    "Florence2Detector: unmatched label '%s' — skipped", raw_label
                )
                continue

            meta = self._label_meta[project_label]
            bbox_orig = self._rescale_bbox(bbox_1000, orig_h, orig_w)

            # Skip degenerate boxes (width or height < 1px after rescaling)
            x1, y1, x2, y2 = bbox_orig
            if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                continue

            detections.append({
                "label":      project_label,
                "class_id":   meta["id"],
                "z_order":    meta["z_order"],
                "confidence": 1.0,   # Florence-2 has no per-box score
                "bbox_xyxy":  bbox_orig,
                "mask":       None,  # SAM2Segmentor fills masks in Stage 3
            })

        logger.debug(
            "Florence2Detector: %d detections on frame (%dx%d)",
            len(detections), orig_w, orig_h,
        )
        return detections

    def _match_label(self, raw_label: str) -> str | None:
        """
        Match a Florence-2 output label string to a project thing-class name.

        Florence-2 tends to echo the prompt word but may capitalise
        differently or return a substring. Strategy:
          1. Exact match (case-insensitive)
          2. Substring match: project label contained in raw_label
          3. No match → None

        Parameters
        ----------
        raw_label : str
            Label string from Florence-2 structured output.

        Returns
        -------
        str | None
            Matched project label name, or None if no match found.
        """
        raw_lower = raw_label.strip().lower()
        # Pass 1: exact
        for proj_label in self._label_meta:
            if proj_label.lower() == raw_lower:
                return proj_label
        # Pass 2: substring (e.g. "small vessel" → "Vessel")
        for proj_label in self._label_meta:
            if proj_label.lower() in raw_lower:
                return proj_label
        return None
