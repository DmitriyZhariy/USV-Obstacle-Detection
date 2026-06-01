"""
SAM 2.1 video tracker for the cpu-sam2 pipeline mode.

Model  : sam2.1_hiera_small  (SAM2VideoPredictor - NOT image predictor)
Task   : Forward propagation of keyframe instance masks across all clip frames.
         Produces sparse keyframe annotations compatible with CVAT Track mode.

Design constraints
-----------------------------------------------------
- Forward propagation only. No backward pass (doubles CPU runtime).
- Keyframe detection : IoU(propagated_mask, last_keyframe_mask) < 0.85
  - mark frame as keyframe=True and update the reference mask.
- Outside detection  : mask_area(mask) < outside_area_threshold (default 100 px^2)
  - emit one final PolygonKeyframe with outside=True, retire the track.
- Every track's first frame is always keyframe=True.
- tqdm progress bar over the frame propagation loop.
- CPU inference; torch.inference_mode() applied per propagation call.
- All paths use pathlib.Path; no POSIX-only constructs.

SAM 2 video predictor API notes
--------------------------------
- `init_state(video_path)` - SAM 2 expects a directory of JPEG/PNG frames
  OR accepts pre-loaded frames via `init_state(video_path, offload_video_to_cpu=True)`.
  We pass frames as a temp directory to avoid re-implementing the loader.
- `add_new_mask(inference_state, frame_idx, obj_id, mask)` - register a
  keyframe mask for one object.
- `propagate_in_video(inference_state)` - yields (frame_idx, obj_ids, masks)
  for every frame in forward order.
- `reset_state(inference_state)` - must be called between clips to free memory.

Output contract
---------------
Returns list[TrackAnnotation] - one entry per tracked instance.
Each TrackAnnotation.keyframes is a sparse list: only frames where
keyframe=True or outside=True are included. CVAT interpolates between
keyframes automatically.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from usv.auto_annotation.tracker.base import TrackerBase
from usv.auto_annotation.types import PolygonKeyframe, TrackAnnotation
from usv.auto_annotation.postprocess.mask_utils import (
    mask_area,
    mask_iou,
    mask_to_polygon,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_KEYFRAME_IOU_THRESH = 0.85
_DEFAULT_OUTSIDE_AREA_THRESH = 100  # px^2
_JPEG_QUALITY = 95                  # temp frame quality; high to avoid artefacts


class SAM2Tracker(TrackerBase):
    """
    Propagates keyframe instance masks across all frames of a clip using
    SAM2VideoPredictor (hiera_small, forward pass only).

    Usage (inside pipeline.py)
        tracker = SAM2Tracker(checkpoint, model_cfg)
        tracker.init_clip(frames, keyframe_idx, instance_masks)
        track_annotations = tracker.finalize()

    Parameters
    checkpoint : Path | str
        Path to sam2.1_hiera_small.pt checkpoint.
    model_cfg : str
        Config name passed to build_sam2(). Must match the checkpoint.
        E.g. "sam2.1_hiera_s"  - do NOT pass a file path.
    keyframe_iou_thresh : float
        IoU drop threshold below which a propagated frame is marked
        keyframe=True and the reference mask is updated. Default 0.85.
    outside_area_thresh : int
        Mask area (px^2) below which a track is considered outside-frame
        and retired. Default 100.
    """

    def __init__(
        self,
        checkpoint: Path | str,
        model_cfg: str = "sam2.1_hiera_s",
        keyframe_iou_thresh: float = _DEFAULT_KEYFRAME_IOU_THRESH,
        outside_area_thresh: int = _DEFAULT_OUTSIDE_AREA_THRESH,
    ) -> None:
        from sam2.build_sam import build_sam2_video_predictor
        import torch

        import os
        import sam2
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        logger.info(
            "SAM2Tracker: loading %s from %s (CPU) ...", model_cfg, checkpoint
        )

        # Point Hydra directly at the sam2 package directory where YAMLs live
        sam2_cfg_dir = os.path.join(os.path.dirname(sam2.__file__), "configs", "sam2.1")
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=sam2_cfg_dir, job_name="sam2", version_base=None)

        self._predictor = build_sam2_video_predictor(
            config_file=model_cfg,
            ckpt_path=str(checkpoint),
            device="cpu",
            apply_postprocessing=False,
        )
        GlobalHydra.instance().clear()
        
        _patch_predictor_for_cpu_fp32(self._predictor)

        self._torch = torch
        self._keyframe_iou_thresh = keyframe_iou_thresh
        self._outside_area_thresh = outside_area_thresh

        # State set by init_clip(); cleared by finalize()
        self._frames: list[np.ndarray] = []
        self._keyframe_idx: int = 0
        self._instance_masks: list[dict] = []   # from SAM2Segmentor Stage 3
        logger.info("SAM2Tracker: ready.")

    # Public initialisation - called once per clip

    def init_clip(
        self,
        frames: list[np.ndarray],
        keyframe_idx: int,
        instance_masks: list[dict],
    ) -> None:
        """
        Register clip frames and keyframe instance masks before propagation.

        Parameters
        ----------
        frames : list[np.ndarray]
            All clip frames in order, BGR uint8. Index 0 = first frame.
        keyframe_idx : int
            Index into `frames` that corresponds to the detection keyframe.
            Instance masks were generated on this frame.
        instance_masks : list[dict]
            Output from SAM2Segmentor.segment() enriched by pipeline.py with
            track metadata. Each dict must contain:
                track_id  : int
                label     : str
                z_order   : int
                mask      : np.ndarray  H×W uint8 binary
        """
        self._frames = frames
        self._keyframe_idx = keyframe_idx
        self._instance_masks = instance_masks

    # TrackerBase interface - update() is a no-op for video predictor

    def update(self, frame_idx: int, detections: list[dict]) -> None:
        """Not used by SAM2Tracker. Propagation is batch, not per-frame."""
        pass

    def finalize(self) -> list[TrackAnnotation]:
        """
        Run SAM 2 forward propagation and build TrackAnnotation list.

        Must be called after init_clip(). Handles its own temp directory
        lifecycle and resets SAM 2 inference state on exit.

        Returns
        -------
        list[TrackAnnotation]
            Sparse keyframe annotations, ready for cvat_xml_exporter.py.
            Returns empty list if no instance masks were registered.
        """
        if not self._instance_masks:
            logger.warning("SAM2Tracker.finalize(): no instance masks - skipping.")
            return []

        if not self._frames:
            raise RuntimeError(
                "SAM2Tracker.finalize() called before init_clip()."
            )

        with tempfile.TemporaryDirectory(prefix="sam2_frames_") as tmp_dir:
            frame_dir = Path(tmp_dir)
            self._write_frames_to_disk(frame_dir)
            track_annotations = self._run_propagation(frame_dir)

        # Clear state so the tracker can be reused for the next clip
        self._frames = []
        self._instance_masks = []
        return track_annotations

    # Internal helpers

    def _write_frames_to_disk(self, frame_dir: Path) -> None:
        """
        Write clip frames as JPEG to a temp directory.

        SAM 2 VideoPredictor expects a directory of image files.
        File names must be zero-padded integers so SAM 2 sorts them
        correctly: 00000.jpg, 00001.jpg, ...
        """
        logger.debug(
            "SAM2Tracker: writing %d frames to %s ...",
            len(self._frames), frame_dir,
        )
        for i, frame_bgr in enumerate(self._frames):
            out_path = frame_dir / f"{i:05d}.jpg"
            cv2.imwrite(
                str(out_path),
                frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
            )

    def _run_propagation(self, frame_dir: Path) -> list[TrackAnnotation]:
        """
        Core propagation loop.

        1. Build SAM 2 inference state from temp frame directory.
        2. Register each instance mask on the keyframe.
        3. Forward-propagate; collect per-frame per-object masks.
        4. Apply keyframe detection and outside detection per object.
        5. Convert masks to polygons; build TrackAnnotation objects.
        """
        from tqdm import tqdm

        torch = self._torch
        n_frames = len(self._frames)

        # Init inference state
        logger.info(
            "SAM2Tracker: initialising inference state (%d frames) ...",
            n_frames,
        )

        with torch.inference_mode():
            # Stage A: build state
            inference_state = self._predictor.init_state(
                video_path=str(frame_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,
            )

            # Stage B: register keyframe masks
            logger.info(
                "SAM2Tracker: registering %d keyframe masks at frame %d ...",
                len(self._instance_masks), self._keyframe_idx,
            )
            for inst in self._instance_masks:
                obj_id: int = inst["track_id"]
                mask: np.ndarray = inst["mask"]
                self._predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=self._keyframe_idx,
                    obj_id=obj_id,
                    mask=mask.astype(bool),
                )

            # Stage C: forward propagation
            logger.info("SAM2Tracker: propagating forward ...")
            meta: dict[int, dict] = {
                inst["track_id"]: {
                    "label":   inst["label"],
                    "z_order": inst["z_order"],
                }
                for inst in self._instance_masks
            }
            raw_frames: dict[int, dict[int, np.ndarray]] = {
                obj_id: {} for obj_id in meta
            }

            propagation_iter = self._predictor.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=n_frames,
                reverse=False,
            )
            for frame_idx, obj_ids, mask_logits in tqdm(
                propagation_iter,
                total=n_frames,
                desc="SAM2 propagation",
                unit="frame",
            ):
                binary_masks = (mask_logits > 0.0).squeeze(1).cpu().float().numpy()
                for i, obj_id in enumerate(obj_ids):
                    if obj_id in raw_frames:
                        raw_frames[obj_id][frame_idx] = (
                            binary_masks[i].astype(np.uint8)
                        )

        # Reset after exiting inference_mode (reset_state is safe outside)
        self._predictor.reset_state(inference_state)

        # Polygon conversion + TrackAnnotation build
        track_annotations: list[TrackAnnotation] = []

        for obj_id, frame_mask_map in raw_frames.items():
            if not frame_mask_map:
                logger.warning(
                    "SAM2Tracker: obj_id=%d has no propagated frames - skipped.",
                    obj_id,
                )
                continue
            keyframes = self._build_keyframes(obj_id, frame_mask_map)
            if not keyframes:
                continue
            track_annotations.append(
                TrackAnnotation(
                    track_id=obj_id,
                    label=meta[obj_id]["label"],
                    z_order=meta[obj_id]["z_order"],
                    keyframes=keyframes,
                )
            )

        logger.info(
            "SAM2Tracker: produced %d track annotations.", len(track_annotations)
        )
        return track_annotations

    def _build_keyframes(
        self,
        obj_id: int,
        frame_mask_map: dict[int, np.ndarray],
    ) -> list[PolygonKeyframe]:
        """
        Convert per-frame masks for one object into a sparse keyframe list.

        Logic
        - First visible frame  - always keyframe=True.
        - Subsequent frames    - keyframe=True if IoU vs. last reference < thresh.
        - If mask_area < outside_thresh - emit outside=True, retire (stop loop).
        - Polygon is always computed (needed even for outside=True final frame).
        - Frames that are not keyframes are skipped (CVAT interpolates between).
        """
        sorted_frames = sorted(frame_mask_map.keys())
        keyframe_list: list[PolygonKeyframe] = []
        last_kf_mask: np.ndarray | None = None

        for frame_idx in sorted_frames:
            mask = frame_mask_map[frame_idx]
            area = mask_area(mask)

            if area < self._outside_area_thresh:
                # only emit outside kf if the object was visible before
                if last_kf_mask is None:
                    # Never had a single visible frame - discard entirely.
                    # Emitting a track that starts with outside=True is invalid
                    # in CVAT: every track must have at least one non-outside kf.
                    logger.debug(
                        "SAM2Tracker: obj_id=%d discarded — "
                        "first propagated frame (idx=%d) already outside "
                        "(area=%d px^2 < threshold=%d px^2).",
                        obj_id, frame_idx, area, self._outside_area_thresh,
                    )
                    return []   # caller skips this track entirely

                # Object was visible before — safe to close the track with outside=True
                polygon = self._safe_polygon(mask, frame_idx, obj_id)
                keyframe_list.append(
                    PolygonKeyframe(
                        frame_idx=frame_idx,
                        points=polygon,
                        keyframe=True,
                        outside=True,
                        occluded=False,
                    )
                )
                logger.debug(
                    "SAM2Tracker: obj_id=%d retired at frame %d "
                    "(area=%d px^2 < threshold=%d px^2).",
                    obj_id, frame_idx, area, self._outside_area_thresh,
                )
                break

            is_first = (last_kf_mask is None)
            is_keyframe = is_first or (mask_iou(mask, last_kf_mask) < self._keyframe_iou_thresh)

            if is_keyframe:
                polygon = self._safe_polygon(mask, frame_idx, obj_id)
                keyframe_list.append(
                    PolygonKeyframe(
                        frame_idx=frame_idx,
                        points=polygon,
                        keyframe=True,
                        outside=False,
                        occluded=False,
                    )
                )
                last_kf_mask = mask

        return keyframe_list

    def _safe_polygon(
        self,
        mask: np.ndarray,
        frame_idx: int,
        obj_id: int,
    ) -> list[tuple[float, float]]:
        """
        Convert mask to polygon. Falls back to bounding-box rectangle
        if mask_to_polygon() returns fewer than 3 points (degenerate shape).
        """
        polygon = mask_to_polygon(mask)
        if polygon is not None and len(polygon) >= 3:
            return polygon

        # Fallback: bounding box from mask extent
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            # Truly empty - return a 1px degenerate polygon at origin
            logger.warning(
                "SAM2Tracker: obj_id=%d frame=%d has empty mask; "
                "using 1px fallback polygon.",
                obj_id, frame_idx,
            )
            return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()), float(ys.max())
        logger.debug(
            "SAM2Tracker: obj_id=%d frame=%d using bbox fallback polygon.",
            obj_id, frame_idx,
        )
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    

def _patch_predictor_for_cpu_fp32(predictor) -> None:
    """
    Patch SAM2's _prepare_memory_conditioned_features to cast maskmem_features
    to float32 at the READ site, immediately before they enter memory_attention.

    Root cause
    ----------
    sam2_video_predictor.py unconditionally casts maskmem_features to bfloat16
    when storing them into the internal output_dict (lines 781 & 833).
    On CPU inference the model weights stay float32, so feeding bf16 features
    into memory_attention raises:
        RuntimeError: mat1 and mat2 must have the same dtype, got BFloat16 and Float

    Why patching _run_single_frame_inference (return value) doesn't work
    ---------------------------------------------------------------------
    That method returns compact_out for the CALLER's storage, but internally
    it also writes into output_dict BEFORE assembling compact_out.
    _prepare_memory_conditioned_features reads from output_dict, so by the
    time our return-value patch fires, the bf16 features are already cached
    inside the model state and will be read back as bf16 on the next frame.

    Fix
    ---
    Patch _prepare_memory_conditioned_features directly on the model (sam2_base)
    to cast maskmem_features to the compute dtype (float32) at the point where
    they are loaded from the cache: the `.to(device)` call on line 575.
    This is the single chokepoint all stored features pass through before
    entering memory_attention, regardless of which code path wrote them.
    """
    import torch
    import types

    model = predictor.model if hasattr(predictor, "model") else predictor

    original_pmcf = model._prepare_memory_conditioned_features.__func__

    def _patched_pmcf(self_inner, *args, **kwargs):
        # Temporarily monkey-patch the stored maskmem_features in every cached
        # output to be float32 before the original method reads them.
        # We do this by wrapping the dict __getitem__ — but that is fragile.
        # Instead: run the original, catch the dtype error, and re-run after
        # casting all cached features. Simpler: just cast before calling.
        #
        # The cleanest approach: patch at the exact line that does
        #   feats = prev["maskmem_features"].to(device, non_blocking=True)
        # by casting all prev["maskmem_features"] in the output_dicts to f32
        # before the original function iterates over them.
        #
        # output_dict is passed as a kwarg or positional arg — find it.
        output_dict = kwargs.get("output_dict") or (args[3] if len(args) > 3 else None)
        if output_dict is not None:
            for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                for frame_out in output_dict.get(storage_key, {}).values():
                    if (
                        isinstance(frame_out, dict)
                        and frame_out.get("maskmem_features") is not None
                    ):
                        frame_out["maskmem_features"] = frame_out[
                            "maskmem_features"
                        ].to(torch.float32)
        return original_pmcf(self_inner, *args, **kwargs)

    model._prepare_memory_conditioned_features = types.MethodType(
        _patched_pmcf, model
    )
    logger.debug(
        "SAM2Tracker: bfloat16→float32 patch applied to "
        "_prepare_memory_conditioned_features."
    )