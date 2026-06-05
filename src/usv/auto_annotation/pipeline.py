"""
AutoAnnotationPipeline — cpu-sam2 mode orchestrator.

Wires Stages 1–8 for the full panoptic pipeline:

    Stage 1  Load clip                   clip_loader.py
    Stage 2  Detect thing instances      Florence2Detector (keyframe only)
    Stage 3  Segment keyframe masks      SAM2Segmentor     (box-prompted)
    Stage 4  Propagate masks             SAM2Tracker       (forward pass)
    Stage 5  Segment stuff per frame     SegFormerStuff    (all frames)
    Stage 6  Convert stuff to polygons   mask_utils.mask_to_polygon
    Stage 7  Merge + resolve overlaps    overlap_resolver.resolve_overlaps
    Stage 8  Export CVAT Video XML       cvat_xml_exporter.export_clip

Design constraints
- cpu-fast path is NOT touched; pipeline.py is additive only.
- CPU inference optimisations applied once in __init__:
      torch.set_num_threads(os.cpu_count())
      torch.set_num_interop_threads(2)
  All model calls wrapped in torch.inference_mode() inside each module.
- Forward propagation only (SAM2Tracker). No backward pass.
- --inference-resize 640: frames resized before Florence-2; bboxes
  rescaled back before SAM2 prompting (handled inside Florence2Detector).
- Keyframe IoU threshold 0.85, outside area threshold 100 px^2 (SAM2Tracker).
- Manifest JSON written alongside CVAT XML.
- Per-stage wall-clock timings logged and stored in manifest.
- Idempotent: honours skip_existing flag from caller.
- All paths via pathlib.Path; no POSIX-only constructs.

Model checkpoints expected layout
    models/
        sam2.1_hiera_small.pt

Florence-2 and SegFormer are downloaded from HuggingFace on first run
and cached in the HF_HOME cache directory (default ~/.cache/huggingface).

Install requirements (flag before first run)
    pip install transformers accelerate tqdm
    pip install git+https://github.com/facebookresearch/segment-anything-2
    # Then download SAM 2 checkpoint:
    # wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt -P models/
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from usv.auto_annotation.types import ClipData, TrackAnnotation

logger = logging.getLogger(__name__)

_SAM2_CHECKPOINT_DEFAULT = "models/sam2.1_hiera_small.pt"
_SAM2_MODEL_CFG          = "sam2.1_hiera_s"
_FLORENCE2_MODEL         = "microsoft/Florence-2-base"
_SEGFORMER_MODEL         = "nvidia/segformer-b0-finetuned-ade-512-512"


class AutoAnnotationPipeline:
    """
    Full panoptic auto-annotation pipeline (cpu-sam2 mode).

    Instantiate once per run (models are loaded in __init__).
    Call run_clip() for each clip to process.

    Parameters
    config_path : Path
        Path to configs/auto_annotation.yaml.
    output_dir : Path
        Root output directory. CVAT exports go to output_dir/cvat_export/.
    inference_resize : int
        Long-side resize in pixels for Florence-2 inference. Default 640.
    keyframe_iou_thresh : float
        IoU drop threshold for SAM2Tracker keyframe detection. Default 0.85.
    outside_area_thresh : int
        Mask area (px^2) below which SAM2Tracker retires a track. Default 100.
    sam2_checkpoint : Path | str
        Path to sam2.1_hiera_small.pt. Default: models/sam2.1_hiera_small.pt.
    skip_existing : bool
        Skip clips whose output zip already exists. Default True.
    debug_vis : bool
        Save coloured overlay JPEGs to output_dir/debug/<clip_name>/. Default False.
    """

    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
        annot_mode: str = "panoptic", # panoptic | instance | semantic
        inference_resize: int = 640,
        keyframe_iou_thresh: float = 0.85,
        outside_area_thresh: int = 100,
        sam2_checkpoint: Path | str = _SAM2_CHECKPOINT_DEFAULT,
        skip_existing: bool = True,
        debug_vis: bool = False,
    ) -> None:
        import torch

        # CPU optimisations
        n_threads = os.cpu_count() or 4
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(2)
        logger.info(
            "AutoAnnotationPipeline: CPU threads = %d (interop = 2)", n_threads
        )

        self._config_path     = Path(config_path)
        self._output_dir      = Path(output_dir)
        self._skip_existing   = skip_existing
        self._debug_vis       = debug_vis
        self._inference_resize = inference_resize
        self._annot_mode = annot_mode

        with open(self._config_path, encoding="utf-8") as f:
            self._cfg: dict[str, Any] = yaml.safe_load(f)

        # Load models (once; reused across all clips)
        logger.info("AutoAnnotationPipeline: loading models (annot_mode=%s) ...",
                    annot_mode)

        
        from usv.auto_annotation.stuff.segformer_stuff import SegFormerStuff
        self._stuff_segmentor = None
        if self._annot_mode in ("panoptic", "semantic"):
            self._stuff_segmentor = SegFormerStuff(
                model_name=_SEGFORMER_MODEL,
            )

        if annot_mode != 'semantic':
            # Validate SAM 2 checkpoint
            self._sam2_checkpoint = Path(sam2_checkpoint)
            if not self._sam2_checkpoint.exists():
                raise FileNotFoundError(
                    f"SAM 2 checkpoint not found: {self._sam2_checkpoint}\n"
                    f"Download with:\n"
                    f"  wget https://dl.fbaipublicfiles.com/segment_anything_2"
                    f"/092824/sam2.1_hiera_small.pt -P models/"
                )
            
            from usv.auto_annotation.detectors.florence2_detector import Florence2Detector
            from usv.auto_annotation.segmentors.sam2_segmentor import SAM2Segmentor
            from usv.auto_annotation.tracker.sam2_tracker import SAM2Tracker
            self._detector = Florence2Detector(
                config_path=self._config_path,
                model_name=_FLORENCE2_MODEL,
                inference_resize=inference_resize,
            )
            self._segmentor = SAM2Segmentor(
                checkpoint=self._sam2_checkpoint,
                model_cfg=_SAM2_MODEL_CFG,
            )
            self._tracker = SAM2Tracker(
                checkpoint=self._sam2_checkpoint,
                model_cfg=_SAM2_MODEL_CFG,
                keyframe_iou_thresh=keyframe_iou_thresh,
                outside_area_thresh=outside_area_thresh,
            )
        else:
            # В semantic режиме SAM2 checkpoint не нужен — не валидируем
            self._detector  = None  # type: ignore[assignment]
            self._segmentor = None
            self._tracker   = None
            logger.info(
                "AutoAnnotationPipeline: semantic mode — "
                "Florence-2 and SAM2 skipped."
            )
        
        logger.info("AutoAnnotationPipeline: all models loaded.")

    # Public entry point

    def run_clip(self, clip_data: ClipData) -> Path | None:
        clip_name = clip_data.clip_name

        # skip_existing: проверяем правильный файл/директорию для каждого режима
        if self._annot_mode == "instance":
            out_file = self._output_dir / "cvat_export" / f"{clip_name}_coco.json"
        elif self._annot_mode == "semantic":
            out_file = self._output_dir / "label_maps" / clip_name
        else:
            out_file = self._output_dir / "cvat_export" / f"{clip_name}.zip"

        if self._skip_existing and out_file.exists():
            logger.info("[SKIP] %s — output already exists", clip_name)
            return None

        logger.info("[START] %s  annot_mode=%s", clip_name, self._annot_mode)
        t_total = time.perf_counter()
        timings: dict[str, float] = {}

        # Stage 2–4: только для instance и panoptic
        if self._annot_mode != "semantic":
            t = time.perf_counter()
            raw_detections = self._stage2_detect(clip_data)
            timings["s2_detect_ms"] = round((time.perf_counter() - t) * 1000)
            logger.info(
                "  S2 detect: %d detections on keyframe %d  (%.0f ms)",
                len(raw_detections), clip_data.keyframe_idx, timings["s2_detect_ms"],
            )
            self._save_raw_detections(clip_data, raw_detections)

            t = time.perf_counter()
            instance_masks = self._stage3_segment(clip_data, raw_detections)
            timings["s3_segment_ms"] = round((time.perf_counter() - t) * 1000)
            logger.info(
                "  S3 segment: %d valid masks  (%.0f ms)",
                len(instance_masks), timings["s3_segment_ms"],
            )

            t = time.perf_counter()
            thing_tracks = self._stage4_track(clip_data, instance_masks)
            timings["s4_track_ms"] = round((time.perf_counter() - t) * 1000)
            logger.info(
                "  S4 track: %d thing tracks  (%.0f ms)",
                len(thing_tracks), timings["s4_track_ms"],
            )
        else:
            raw_detections = []
            instance_masks = []
            thing_tracks   = []
            timings["s2_detect_ms"] = timings["s3_segment_ms"] = timings["s4_track_ms"] = 0

        # Stage 5: Stuff segmentation — нужна в panoptic и semantic
        if self._annot_mode in ("panoptic", "semantic"):
            t = time.perf_counter()
            stuff_maps = self._stage5_stuff(clip_data)
            timings["s5_stuff_ms"] = round((time.perf_counter() - t) * 1000)
            logger.info(
                "  S5 stuff: %d frames segmented  (%.0f ms)",
                len(stuff_maps), timings["s5_stuff_ms"],
            )
        else:
            stuff_maps = []
            timings["s5_stuff_ms"] = 0

        # Stage 7: Overlap resolution — только panoptic
        if self._annot_mode == "panoptic":
            t = time.perf_counter()
            instance_meta = _build_instance_meta(instance_masks)
            resolved_things, stuff_tracks = self._stage7_resolve(
                stuff_maps, thing_tracks, instance_meta
            )
            timings["s7_resolve_ms"] = round((time.perf_counter() - t) * 1000)
            logger.info(
                "  S7 resolve: %d thing tracks, %d stuff tracks  (%.0f ms)",
                len(resolved_things), len(stuff_tracks), timings["s7_resolve_ms"],
            )
        else:
            resolved_things = thing_tracks
            stuff_tracks    = []
            timings["s7_resolve_ms"] = 0

        # Stage 8/9: Export
        t = time.perf_counter()
        if self._annot_mode == "instance":
            zip_path   = self._stage8_export_coco(clip_data, resolved_things)
            all_tracks = resolved_things
        elif self._annot_mode == "semantic":
            zip_path   = self._stage8_export_label_maps(clip_data, stuff_maps)
            all_tracks = []
        else:  # panoptic
            all_tracks = stuff_tracks + resolved_things
            zip_path   = self._stage8_export(clip_data, all_tracks)
        timings["s8_export_ms"] = round((time.perf_counter() - t) * 1000)

        timings["total_ms"] = round((time.perf_counter() - t_total) * 1000)
        logger.info(
            "  [DONE] %s → %s  (total %.1f s)",
            clip_name, zip_path, timings["total_ms"] / 1000,
        )

        self._write_manifest(clip_data, all_tracks, timings)

        if self._debug_vis and instance_masks:
            self._save_debug_vis(clip_data, instance_masks)

        return zip_path

    # Stage implementations

    def _stage2_detect(self, clip_data: ClipData) -> list[dict]:
        """
        Stage 2: Florence-2 detection on the keyframe only.

        Falls back to adjacent frames if zero detections on keyframe,
        as specified in design doc §4 Stage 2 failure modes.
        """
        keyframe = clip_data.frames[clip_data.keyframe_idx]
        detections = self._detector.detect(keyframe)

        if not detections:
            logger.warning(
                "  S2: zero detections on keyframe %d — trying adjacent frames.",
                clip_data.keyframe_idx,
            )
            for offset in [-1, +1, -2, +2]:
                alt_idx = clip_data.keyframe_idx + offset
                if 0 <= alt_idx < clip_data.n_frames:
                    detections = self._detector.detect(
                        clip_data.frames[alt_idx]
                    )
                    if detections:
                        logger.info(
                            "  S2: found %d detections at adjacent frame %d.",
                            len(detections), alt_idx,
                        )
                        break

        if not detections:
            logger.warning("  S2: no detections found — clip will have 0 tracks.")

        return detections

    def _save_raw_detections(
        self,
        clip_data: ClipData,
        detections: list[dict],
    ) -> None:
        """
        Persist Stage 2 detections to disk so run_tracking_only.py can
        re-run Stages 3–4 without re-invoking Florence-2.

        Output path:
            <output_dir>/raw/<clip_name>/detections.json

        Schema:
            {
            "clip_name":    str,
            "keyframe_idx": int,
            "detections": [
                {
                "label":      str,
                "class_id":   int,
                "z_order":    int,
                "confidence": float,
                "bbox_xyxy":  [x1, y1, x2, y2]
                // mask intentionally omitted — not JSON-serialisable,
                // and is re-generated by SAM2Segmentor in Stage 3
                },
                ...
            ]
            }

        bbox_xyxy values are cast to float so json.dump never chokes on
        numpy scalars.
        """
        raw_dir = self._output_dir / "raw" / clip_data.clip_name
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_path = raw_dir / "detections.json"

        serialisable = [
            {
                "label":      det["label"],
                "class_id":   int(det["class_id"]),
                "z_order":    int(det["z_order"]),
                "confidence": float(det["confidence"]),
                "bbox_xyxy":  [float(v) for v in det["bbox_xyxy"]],
            }
            for det in detections
        ]

        payload = {
            "clip_name":    clip_data.clip_name,
            "keyframe_idx": clip_data.keyframe_idx,
            "detections":   serialisable,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(
            "  Raw detections saved: %s  (%d detections)",
            out_path, len(serialisable),
        )

    def _stage3_segment(self, clip_data, detections):
        from usv.auto_annotation.postprocess.mask_utils import mask_area

        if not detections:
            return []

        keyframe = clip_data.frames[clip_data.keyframe_idx]
        h, w = keyframe.shape[:2]
        frame_area = h * w

        # Масштабируемый порог
        frac = float(self._cfg.get("min_instance_area_frac", 0.0))
        if frac > 0.0:
            min_area = int(frac * frame_area)
        else:
            # fallback на абсолютный px² для обратной совместимости
            min_area = int(self._cfg.get("min_instance_area", 64))
        logger.debug(
            "  S3: min_instance_area = %d px² (frame %dx%d, frac=%.4f)",
            min_area, w, h, frac,
        )
        # остальной код без изменений
        instance_masks: list[dict] = []
        track_id = 1

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = bw * 2, bh * 2
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(w, int(x2 + pad_x))
            cy2 = min(h, int(y2 + pad_y))

            crop = keyframe[cy1:cy2, cx1:cx2]
            bbox_in_crop = [[x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]]
            raw_masks_crop = self._segmentor.segment(crop, bbox_in_crop)
            if not raw_masks_crop:
                continue

            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[cy1:cy2, cx1:cx2] = raw_masks_crop[0]

            area = mask_area(full_mask)
            if area < min_area:
                logger.debug(
                    "  S3: discard '%s' — area %d px² < min %d px²",
                    det["label"], area, min_area,
                )
                continue

            instance_masks.append({
                "track_id":   track_id,
                "label":      det["label"],
                "class_id":   det["class_id"],
                "z_order":    det["z_order"],
                "confidence": det["confidence"],
                "bbox_xyxy":  det["bbox_xyxy"],
                "mask":       full_mask,
            })
            track_id += 1

        logger.debug("  S3: %d / %d instances passed area filter.",
                    len(instance_masks), len(detections))
        return instance_masks

    def _stage4_track(
        self,
        clip_data: ClipData,
        instance_masks: list[dict],
    ) -> list[TrackAnnotation]:
        """
        Stage 4: SAM2Tracker forward propagation across all clip frames.
        Returns empty list if no valid instance masks from Stage 3.
        """
        if not instance_masks:
            return []

        self._tracker.init_clip(
            frames=clip_data.frames,
            keyframe_idx=clip_data.keyframe_idx,
            instance_masks=instance_masks,
        )
        return self._tracker.finalize()

    def _stage5_stuff(self, clip_data: ClipData) -> list[np.ndarray]:
        """
        Stage 5: SegFormerStuff on all frames.
        Returns list of H×W uint8 project class_id maps.
        """
        return [self._stuff_segmentor.segment_frame(f) for f in clip_data.frames]

    def _stage7_resolve(
        self,
        stuff_maps: list[np.ndarray],
        thing_tracks: list[TrackAnnotation],
        instance_meta: dict[int, dict],
    ) -> tuple[list[TrackAnnotation], list[TrackAnnotation]]:
        """
        Stage 7: Merge stuff maps + thing tracks via overlap_resolver.
        (Stage 6 — polygon conversion — is handled inside overlap_resolver.)
        """
        from usv.auto_annotation.postprocess.overlap_resolver import resolve_overlaps

        return resolve_overlaps(
            stuff_maps=stuff_maps,
            thing_tracks=thing_tracks,
            instance_meta=instance_meta,
            config=self._cfg,
        )

    def _stage8_export(
        self,
        clip_data: ClipData,
        all_tracks: list[TrackAnnotation],
    ) -> Path:
        """
        Stage 8: Serialise TrackAnnotation list to CVAT Video XML zip.
        """
        from usv.auto_annotation.exporters.cvat_xml_exporter import export_clip

        out_dir = self._output_dir / "cvat_export"
        return export_clip(
            clip_data=clip_data,
            tracks=all_tracks,
            output_dir=out_dir,
            write_zip=True,
        )
    
    def _stage8_export_coco(
        self,
        clip_data: ClipData,
        all_tracks: list[TrackAnnotation],
    ) -> Path:
        """
        Stage 8: Serialise TrackAnnotation list to CVAT Video XML zip.
        """
        from usv.auto_annotation.exporters.coco_exporter import export_coco

        out_dir = self._output_dir / "cvat_export"
        return export_coco(
            clip_data=clip_data,
            tracks=all_tracks,
            output_dir=out_dir,
        )
    
    def _stage8_export_label_maps(
        self,
        clip_data: ClipData,
        label_maps: list[np.ndarray],
    ) -> Path:
        """
        Stage 9: Export per-frame PNG label maps (semantic mode only).
        """
        from usv.auto_annotation.exporters.label_map_exporter import export_label_maps

        out_dir = self._output_dir / "label_maps"
        return export_label_maps(
            clip_name=clip_data.clip_name,
            label_maps=label_maps,
            output_dir=out_dir,
            save_vis=self._debug_vis,  # vis только если --debug-vis
        )

    # Manifest + debug helpers

    def _write_manifest(
        self,
        clip_data: ClipData,
        all_tracks: list[TrackAnnotation],
        timings: dict[str, float],
    ) -> None:
        """Write per-clip manifest JSON alongside the CVAT export zip."""
        manifest = {
            "clip_name":    clip_data.clip_name,
            "mode":         "cpu-sam2",
            "n_frames":     clip_data.n_frames,
            "keyframe_idx": clip_data.keyframe_idx,
            "n_tracks":     len(all_tracks),
            "track_labels": [t.label for t in all_tracks],
            "timings_ms":   timings,
        }
        manifest_dir = self._output_dir / "cvat_export"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{clip_data.clip_name}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info("  Manifest written: %s", manifest_path)

    def _save_debug_vis(
        self,
        clip_data: ClipData,
        instance_masks: list[dict],
    ) -> None:
        """
        Save keyframe debug overlay JPEG showing instance masks + labels.
        Only the keyframe is visualised (masks are only on the keyframe
        at Stage 3 — propagated frames are sparse polygons, not stored).
        """
        import cv2

        debug_dir = self._output_dir / "debug" / clip_data.clip_name
        debug_dir.mkdir(parents=True, exist_ok=True)

        vis = clip_data.frames[clip_data.keyframe_idx].copy()
        colours = [
            (0, 255, 0), (255, 128, 0), (0, 128, 255),
            (255, 0, 128), (128, 0, 255), (0, 255, 255),
        ]
        for i, inst in enumerate(instance_masks):
            colour = colours[i % len(colours)]
            mask = inst["mask"]
            overlay = vis.copy()
            overlay[mask > 0] = colour
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
            x1, y1, x2, y2 = (int(v) for v in inst["bbox_xyxy"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                vis,
                f"{inst['label']} #{inst['track_id']}",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
            )

        out_path = debug_dir / f"keyframe_{clip_data.keyframe_idx:04d}_s3.jpg"
        cv2.imwrite(str(out_path), vis)
        logger.info("  Debug vis saved: %s", out_path)

# Internal helpers

def _build_instance_meta(instance_masks: list[dict]) -> dict[int, dict]:
    """
    Build instance_meta dict for overlap_resolver from Stage 3 output.
    track_id → {label, z_order, confidence}
    """
    return {
        inst["track_id"]: {
            "label":      inst["label"],
            "z_order":    inst["z_order"],
            "confidence": inst.get("confidence", 1.0),
        }
        for inst in instance_masks
    }
