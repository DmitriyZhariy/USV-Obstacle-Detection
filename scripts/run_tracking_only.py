"""
Re-run SAM2Segmentor and SAM2Tracker without re-running Florence-2.

Reads raw per-frame detection JSON written by pipeline.py Stage 2, runs
SAM2Segmentor on the keyframe, then SAM2Tracker for propagation.
Overwrites the tracked JSON and re-exports the CVAT zip.

Use case: you have fixed detection outputs (e.g. manually edited raw JSON)
and want to re-propagate without paying the Florence-2 inference cost.

Usage:
    python -m scripts.run_tracking_only `
        --config configs/auto_annotation.yaml `
        --clips-dir data/interim/choosed_clips_v5-1 `
        --raw-dir data/interim/auto_annotations/raw `
        --output-dir data/interim/auto_annotations `
        --clip-name left_MOVI0017_0001
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SAM2_CHECKPOINT_DEFAULT = "models/sam2.1_hiera_small.pt"
_SAM2_MODEL_CFG = "sam2.1_hiera_small"


class TrackingOnlyRunner:
    """Re-run SAM2 segmentation + tracking from saved raw detections."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.clips_dir  = Path(args.clips_dir)
        self.raw_dir    = Path(args.raw_dir)
        self.output_dir = Path(args.output_dir)
        self.config_path = Path(args.config)
        self.skip_existing = args.skip_existing
        self.sam2_checkpoint = Path(
            args.sam2_checkpoint or _SAM2_CHECKPOINT_DEFAULT
        )

        if not self.sam2_checkpoint.exists():
            logger.error(
                "SAM 2 checkpoint not found: %s", self.sam2_checkpoint
            )
            sys.exit(1)

        with open(self.config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # Load models once
        import os, torch
        torch.set_num_threads(os.cpu_count() or 4)
        torch.set_num_interop_threads(2)

        from usv.auto_annotation.segmentors.sam2_segmentor import SAM2Segmentor
        from usv.auto_annotation.tracker.sam2_tracker import SAM2Tracker

        self._segmentor = SAM2Segmentor(
            checkpoint=self.sam2_checkpoint,
            model_cfg=_SAM2_MODEL_CFG,
        )
        self._tracker = SAM2Tracker(
            checkpoint=self.sam2_checkpoint,
            model_cfg=_SAM2_MODEL_CFG,
            keyframe_iou_thresh=float(args.keyframe_iou_threshold),
            outside_area_thresh=int(
                self.cfg.get("outside_area_threshold", 100)
            ),
        )

    def run(self, clip_name: str | None, process_all: bool) -> None:
        clips = self._resolve_clips(clip_name, process_all)
        logger.info("Clips to re-track: %d", len(clips))
        for name in clips:
            try:
                self._run_one(name)
            except Exception:
                logger.exception("[FAIL] %s", name)

    def _resolve_clips(
        self, clip_name: str | None, process_all: bool
    ) -> list[str]:
        if process_all:
            from usv.auto_annotation.clip_loader import ClipLoader
            return ClipLoader(self.clips_dir).list_clips()
        if clip_name:
            return [clip_name]
        logger.error("Specify --clip-name or --all")
        sys.exit(1)

    def _run_one(self, clip_name: str) -> None:
        from usv.auto_annotation.clip_loader import ClipLoader
        from usv.auto_annotation.exporters.cvat_xml_exporter import export_clip
        from usv.auto_annotation.postprocess.mask_utils import mask_area

        out_zip = self.output_dir / "cvat_export" / f"{clip_name}.zip"
        if self.skip_existing and out_zip.exists():
            logger.info("[SKIP] %s", clip_name)
            return

        t0 = time.perf_counter()
        logger.info("[START] %s", clip_name)

        # Load clip
        clip_data = ClipLoader(self.clips_dir).load(clip_name)

        # Load raw detections saved by pipeline Stage 2
        raw_path = self.raw_dir / clip_name / "detections.json"
        if not raw_path.exists():
            logger.error(
                "Raw detections not found: %s\n"
                "Run cpu-sam2 pipeline first to generate Stage 2 output.",
                raw_path,
            )
            return

        with open(raw_path, encoding="utf-8") as f:
            raw = json.load(f)

        detections: list[dict] = raw.get("detections", [])
        keyframe_idx: int = raw.get("keyframe_idx", clip_data.keyframe_idx)

        if not detections:
            logger.warning("[WARN] %s — no detections in raw JSON, skipping.", clip_name)
            return

        # re-segment keyframe with SAM2
        min_area = int(self.cfg.get("min_instance_area", 64))
        keyframe = clip_data.frames[keyframe_idx]
        bboxes = [det["bbox_xyxy"] for det in detections]
        raw_masks = self._segmentor.segment(keyframe, bboxes)

        instance_masks: list[dict] = []
        track_id = 1
        for det, mask in zip(detections, raw_masks):
            if mask_area(mask) < min_area:
                continue
            instance_masks.append({
                "track_id":   track_id,
                "label":      det["label"],
                "class_id":   det["class_id"],
                "z_order":    det["z_order"],
                "confidence": det.get("confidence", 1.0),
                "bbox_xyxy":  det["bbox_xyxy"],
                "mask":       mask,
            })
            track_id += 1

        logger.info("  %d instances after area filter", len(instance_masks))

        if not instance_masks:
            logger.warning("[WARN] %s — no instances passed area filter.", clip_name)
            return

        # re-propagate with SAM2Tracker
        self._tracker.init_clip(
            frames=clip_data.frames,
            keyframe_idx=keyframe_idx,
            instance_masks=instance_masks,
        )
        tracks = self._tracker.finalize()
        logger.info("  %d tracks produced", len(tracks))

        # re-export CVAT zip
        out_dir = self.output_dir / "cvat_export"
        xml_path = export_clip(
            clip_data=clip_data,
            tracks=tracks,
            output_dir=out_dir,
            write_zip=True,
        )
        logger.info(
            "[DONE] %s → %s  (%.1f s)",
            clip_name, xml_path, time.perf_counter() - t0,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-run SAM2 tracking from saved raw detections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--clips-dir",  default="data/interim/choosed_clips_v5-1")
    p.add_argument("--raw-dir",    default="data/interim/auto_annotations/raw",
                   help="Dir containing per-clip detections.json files.")
    p.add_argument("--output-dir", default="data/interim/auto_annotations")
    p.add_argument("--config",     default="configs/auto_annotation.yaml")
    p.add_argument("--clip-name",  default=None)
    p.add_argument("--all",        action="store_true", dest="all_clips")
    p.add_argument("--sam2-checkpoint", default=None)
    p.add_argument("--keyframe-iou-threshold", default=0.85, type=float)
    p.add_argument("--skip-existing",    action="store_true",  default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    runner = TrackingOnlyRunner(args)
    runner.run(clip_name=args.clip_name, process_all=args.all_clips)
