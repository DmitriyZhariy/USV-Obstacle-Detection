"""
Greedy IoU tracker for the cpu-fast pipeline mode.
Pure NumPy - no model dependencies.
Produces list[TrackAnnotation] using the same dataclasses as SAM2Tracker,
so the exporter and downstream code are mode-agnostic.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np

from usv.auto_annotation.types import TrackAnnotation, PolygonKeyframe
from usv.auto_annotation.postprocess.mask_utils import bbox_iou, mask_to_polygon

from usv.auto_annotation.tracker.base import TrackerBase

logger = logging.getLogger(__name__)

_IOU_THRESHOLD = 0.4
_OUTSIDE_AREA_THRESHOLD = 100  # px^2 - below this, outside=True


@dataclass
class _ActiveTrack:
    track_id: int
    label: str
    z_order: int
    last_bbox: tuple[float, float, float, float]
    last_frame_idx: int
    keyframes: list[PolygonKeyframe] = field(default_factory=list)
    missed_frames: int = 0

class IoUTracker(TrackerBase):
    """
    Greedy IoU tracker: links per-frame detections into TrackAnnotation objects.

    Per frame:
      1. Match active tracks to detections by max bbox IoU (same-label only).
      2. Matched detections - append keyframe polygon to existing track.
      3. Unmatched active tracks - append outside=True keyframe, retire track.
      4. Unmatched detections - start new track.
    """

    def __init__(
        self,
        iou_threshold: float = _IOU_THRESHOLD,
        outside_area_threshold: int = _OUTSIDE_AREA_THRESHOLD,
        polygon_epsilon_ratio: float = 0.005,
        min_instance_area: int = 64,
        max_age: int = 3,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._outside_area = outside_area_threshold
        self._epsilon_ratio = polygon_epsilon_ratio
        self._min_area = min_instance_area
        self._next_id: int = 1
        self._active: list[_ActiveTrack] = []
        self._finished: list[TrackAnnotation] = []
        self._max_age = max_age

    def _make_polygon(self, det: dict) -> list[tuple[float, float]] | None:
        mask: np.ndarray | None = det.get("mask")
        if mask is not None:
            pts = mask_to_polygon(
                mask,
                epsilon_ratio=self._epsilon_ratio,
                min_area=self._min_area,
            )
            if pts is not None:
                return pts
        # Fallback: bounding-box rectangle (always 4 valid points)
        x1, y1, x2, y2 = det["bbox_xyxy"]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def update(self, frame_idx: int, frame: "np.ndarray", detections: list[dict]) -> None:
        """Process one frame of detections, updating active tracks in place."""

        # Seed frame or empty active list — all detections start new tracks
        if not self._active:
            for det in detections:
                pts = self._make_polygon(det)
                if pts is None:
                    continue
                self._active.append(_ActiveTrack(
                    track_id=self._next_id,
                    label=det["label"],
                    z_order=det["z_order"],
                    last_bbox=det["bbox_xyxy"],
                    last_frame_idx=frame_idx,
                    keyframes=[PolygonKeyframe(
                        frame_idx=frame_idx, points=pts, keyframe=True
                    )],
                ))
                self._next_id += 1
            return

        n_tracks = len(self._active)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets), dtype=float)

        for ti, track in enumerate(self._active):
            for di, det in enumerate(detections):
                if track.label != det["label"]:
                    continue  # never cross-match different classes
                iou_matrix[ti, di] = bbox_iou(track.last_bbox, det["bbox_xyxy"])

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # Greedy match: highest IoU pairs first
        for flat_idx in np.argsort(-iou_matrix, axis=None):
            ti, di = divmod(int(flat_idx), n_dets)
            if iou_matrix[ti, di] < self._iou_threshold:
                break
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)

            det = detections[di]
            track = self._active[ti]
            pts = self._make_polygon(det)
            if pts is None:
                continue
            track.keyframes.append(
                PolygonKeyframe(frame_idx=frame_idx, points=pts, keyframe=True)
            )
            track.last_bbox = det["bbox_xyxy"]
            track.last_frame_idx = frame_idx

        # Unmatched active tracks - increment missed counter; retire only after max_age.
        still_active: list[_ActiveTrack] = []
        for ti, track in enumerate(self._active):
            if ti not in matched_tracks:
                track.missed_frames += 1
                if track.missed_frames > self._max_age:
                    if track.keyframes:
                        track.keyframes.append(PolygonKeyframe(
                            frame_idx=frame_idx,
                            points=track.keyframes[-1].points,
                            keyframe=True,
                            outside=True,
                        ))
                    self._finished.append(TrackAnnotation(
                        track_id=track.track_id,
                        label=track.label,
                        z_order=track.z_order,
                        keyframes=track.keyframes,
                    ))
                else:
                    still_active.append(track)  # keep alive during grace period
            else:
                track.missed_frames = 0         # reset on successful match
                still_active.append(track)

        # New tracks from unmatched detections
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            pts = self._make_polygon(det)
            if pts is None:
                continue
            still_active.append(_ActiveTrack(
                track_id=self._next_id,
                label=det["label"],
                z_order=det["z_order"],
                last_bbox=det["bbox_xyxy"],
                last_frame_idx=frame_idx,
                keyframes=[PolygonKeyframe(
                    frame_idx=frame_idx, points=pts, keyframe=True
                )],
            ))
            self._next_id += 1

        self._active = still_active

    def finalize(self) -> list[TrackAnnotation]:
        """
        Close all remaining active tracks at their last seen frame.
        Call once after processing all frames.
        """
        result = list(self._finished)
        for track in self._active:
            if track.keyframes:
                result.append(TrackAnnotation(
                    track_id=track.track_id,
                    label=track.label,
                    z_order=track.z_order,
                    keyframes=track.keyframes,
                ))
        # Discard any zero-keyframe entries (shouldn't occur, but be defensive)
        return [t for t in result if t.keyframes]
