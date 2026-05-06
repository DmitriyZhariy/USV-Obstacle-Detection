"""
CLI entry point for the auto-annotation pipeline (Phase 1: cpu-fast mode).

Usage — single clip:
    python -m scripts.run_auto_annotation `
        --config configs/auto_annotation.yaml `
        --mode cpu-fast `
        --clip-name left_MOVI0017_0001

Usage — all clips, skip existing:
    python -m scripts.run_auto_annotation `
        --config configs/auto_annotation.yaml `
        --mode cpu-fast `
        --all --skip-existing
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
import numpy as np
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class AutoAnnotationRunner:
    """Orchestrates the cpu-fast annotation pipeline for one or more clips."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.clips_dir = Path(args.clips_dir)
        self.output_dir = Path(args.output_dir)
        self.config_path = Path(args.config)
        self.mode = args.mode
        self.skip_existing = args.skip_existing
        self.debug_vis = args.debug_vis

        with open(self.config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.args = args

        if self.mode == "cpu-fast":
            self._init_cpu_fast()
        else:
            raise NotImplementedError(
                f"--mode {self.mode} is not yet implemented."
            )

    def _init_cpu_fast(self) -> None:
        from usv.auto_annotation.detectors.yolov8_detector import YOLOv8Detector
        from usv.auto_annotation.tracker.iou_tracker import IoUTracker

        # Resolve model path: CLI arg - repo root - error
        if self.args.model_path:
            model_path = "models/" / Path(self.args.model_path)
        else:
            model_path = Path(__file__).resolve().parent.parent / "models/yolov8n-seg.pt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"yolov8n-seg.pt not found at {model_path}. "
                f"Place it in the repo root or pass --model-path explicitly."
            )

        self._detector = YOLOv8Detector(
            config_path=self.config_path,
            model_path=str(model_path),
        )
        self._tracker_cls = IoUTracker
        logger.info("cpu-fast mode: YOLOv8n-seg + IoU tracker (model: %s)", model_path)

    def _resolve_clips(self, clip_name: str | None, process_all: bool) -> list[str]:
        if process_all:
            from usv.auto_annotation.clip_loader import ClipLoader
            return ClipLoader(self.clips_dir).list_clips()
        if clip_name:
            return [clip_name]
        logger.error("Specify --clip-name or --all")
        sys.exit(1)

    def _should_skip(self, clip_name: str) -> bool:
        zip_path = self.output_dir / "cvat_export" / f"{clip_name}.zip"
        return self.skip_existing and zip_path.exists()

    def run(self, clip_name: str | None, process_all: bool) -> None:
        clips = self._resolve_clips(clip_name, process_all)
        logger.info("Clips to process: %d", len(clips))
        for name in clips:
            if self._should_skip(name):
                logger.info("[SKIP] %s — output zip already exists", name)
                continue
            try:
                self._run_one(name)
            except Exception:
                logger.exception("[FAIL] %s", name)

    def _run_one(self, clip_name: str) -> None:
        from usv.auto_annotation.clip_loader import ClipLoader
        from usv.auto_annotation.exporters.cvat_xml_exporter import export_clip

        t0 = time.perf_counter()
        logger.info("[START] %s  mode=%s", clip_name, self.mode)

        loader = ClipLoader(clips_dir=self.clips_dir)
        clip_data = loader.load(clip_name)
        logger.info(
            "  Loaded %d frames  keyframe_idx=%d  (%dx%d)",
            clip_data.n_frames,
            clip_data.keyframe_idx,
            clip_data.frame_width,
            clip_data.frame_height,
        )

        tracker = self._tracker_cls(
            iou_threshold=0.4,
            outside_area_threshold=int(self.cfg.get("outside_area_threshold", 100)),
            polygon_epsilon_ratio=float(self.cfg.get("polygon_epsilon_ratio", 0.005)),
            min_instance_area=int(self.cfg.get("min_instance_area", 64)),
        )

        for frame_idx, frame_bgr in enumerate(clip_data.frames):
            dets = self._detector.detect(frame_bgr)
            tracker.update(frame_idx, dets)
            logger.debug(
                "  frame %02d/%02d — %d detection(s)",
                frame_idx + 1, clip_data.n_frames, len(dets),
            )
            if self.debug_vis:
                self._save_debug_frame(clip_name, frame_idx, frame_bgr, dets)

        tracks = tracker.finalize()
        logger.info("  Tracks produced: %d", len(tracks))

        if not tracks:
            logger.warning(
                "  [WARN] No tracks produced for %s. "
                "Check coco_mapping in config — YOLOv8 may not detect any mapped class.",
                clip_name,
            )
            return

        out_dir = self.output_dir / "cvat_export"
        xml_path = export_clip(
            clip_data=clip_data,
            tracks=tracks,
            output_dir=out_dir,
            write_zip=True,
        )
        elapsed = time.perf_counter() - t0
        logger.info("  [DONE] %s - %s  (%.1f s)", clip_name, xml_path, elapsed)

    def _save_debug_frame(
        self,
        clip_name: str,
        frame_idx: int,
        frame_bgr: "np.ndarray",
        dets: list[dict],
    ) -> None:
        import cv2
        debug_dir = self.output_dir / "debug" / clip_name
        debug_dir.mkdir(parents=True, exist_ok=True)
        vis = frame_bgr.copy()
        for det in dets:
            x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{det['label']} {det['confidence']:.2f}",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        cv2.imwrite(str(debug_dir / f"{frame_idx:04d}.jpg"), vis)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run auto-annotation pipeline on USV clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--clips-dir", default="data/interim/choosed_clips_v5-1",
        help="Root dir with frames/ and metadata/ subdirs.",
    )
    p.add_argument(
        "--output-dir", default="data/interim/auto_annotations",
        help="Root output dir for all pipeline artifacts.",
    )
    # in _parse_args(), add:
    p.add_argument("--model-path", default=None,
                help="Explicit path to .pt model file. "
                        "Defaults to yolov8n-seg.pt in repo root.")
    p.add_argument(
        "--config", default="configs/auto_annotation.yaml",
    )
    p.add_argument(
        "--mode", choices=["cpu-fast", "cpu-sam2"], default="cpu-fast",
        help="cpu-fast=YOLOv8+IoU (Phase 1). cpu-sam2=Phase 2 (not yet implemented).",
    )
    p.add_argument("--clip-name", default=None)
    p.add_argument("--all", action="store_true", dest="all_clips",
                   help="Process all clips under --clips-dir/frames/.")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--debug-vis", action="store_true",
                   help="Save detection overlay JPEGs to output-dir/debug/.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    runner = AutoAnnotationRunner(args)
    runner.run(clip_name=args.clip_name, process_all=args.all_clips)
