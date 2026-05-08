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
        E.g. "configs/sam2.1/sam2.1_hiera_s"  - do NOT pass a file path.
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
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_s",
        keyframe_iou_thresh: float = _DEFAULT_KEYFRAME_IOU_THRESH,
        outside_area_thresh: int = _DEFAULT_OUTSIDE_AREA_THRESH,
    ) -> None:
        from sam2.build_sam import build_sam2_video_predictor
        import torch

        logger.info(
            "SAM2Tracker: loading %s from %s (CPU) ...", model_cfg, checkpoint
        )
        self._predictor = build_sam2_video_predictor(
            model_cfg,
            str(checkpoint),
            device="cpu",
        )
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
            inference_state = self._predictor.init_state(
                video_path=str(frame_dir),
                offload_video_to_cpu=True,   # mandatory for CPU-only machines
                offload_state_to_cpu=True,
                async_loading_frames=False,
            )

        # Register keyframe masks for all instances
        logger.info(
            "SAM2Tracker: registering %d keyframe masks at frame %d ...",
            len(self._instance_masks), self._keyframe_idx,
        )
        with torch.inference_mode():
            for inst in self._instance_masks:
                obj_id: int = inst["track_id"]
                mask: np.ndarray = inst["mask"]   # H×W uint8 binary
                self._predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=self._keyframe_idx,
                    obj_id=obj_id,
                    mask=mask.astype(bool),
                )

        # Build per-object accumulators
        # meta[obj_id] = {label, z_order}
        meta: dict[int, dict] = {
            inst["track_id"]: {
                "label":   inst["label"],
                "z_order": inst["z_order"],
            }
            for inst in self._instance_masks
        }

        # raw_frames[obj_id][frame_idx] = H×W uint8 mask
        raw_frames: dict[int, dict[int, np.ndarray]] = {
            obj_id: {} for obj_id in meta
        }

        # Forward propagation
        logger.info("SAM2Tracker: propagating forward ...")
        with torch.inference_mode():
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
                # mask_logits : (N_obj, 1, H, W) float32
                # Threshold at 0 to get binary mask (SAM 2 convention)
                binary_masks = (mask_logits > 0.0).squeeze(1).cpu().numpy()
                # binary_masks : (N_obj, H, W) bool
                for i, obj_id in enumerate(obj_ids):
                    if obj_id in raw_frames:
                        raw_frames[obj_id][frame_idx] = (
                            binary_masks[i].astype(np.uint8)
                        )

        # Reset SAM 2 state to free CPU memory before polygon conversion
        self._predictor.reset_state(inference_state)

        # Keyframe detection + outside detection + polygon build
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

            # Outside / retired
            if area < self._outside_area_thresh:
                polygon = self._safe_polygon(mask, frame_idx, obj_id)
                keyframe_list.append(
                    PolygonKeyframe(
                        frame_idx=frame_idx,
                        points=polygon,
                        keyframe=True,   # outside frames are always keyframes
                        outside=True,
                        occluded=False,
                    )
                )
                logger.debug(
                    "SAM2Tracker: obj_id=%d retired at frame %d "
                    "(area=%d px^2 < threshold=%d px^2)",
                    obj_id, frame_idx, area, self._outside_area_thresh,
                )
                break  # retire: no further frames for this track

            is_first = (last_kf_mask is None)

            if is_first:
                is_keyframe = True
            else:
                iou = mask_iou(mask, last_kf_mask)
                is_keyframe = iou < self._keyframe_iou_thresh

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
                last_kf_mask = mask   # update reference mask

            # Non-keyframe frames: skip - CVAT interpolates between keyframes

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
        if len(polygon) >= 3:
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
