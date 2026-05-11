"""
Overlap resolver for the cpu-sam2 pipeline mode.

Responsibility

Merge per-frame stuff segmentation maps (Stage 5) and per-frame thing
instance masks (Stage 4) into a unified list of TrackAnnotation objects
with correct Z-order, ready for cvat_xml_exporter.py.

Rules

1. Stuff polygons form the base layer at their assigned Z-order.
   Each stuff class becomes ONE track spanning the full clip.
   Only frames where the polygon IoU vs. the previous keyframe drops
   below STUFF_KEYFRAME_IOU_THRESH (0.90) are marked keyframe=True.
   The first frame of each stuff class is always keyframe=True.

2. Thing instance masks are drawn at their class Z-order, on top of stuff.
   Thing tracks are passed through unchanged - they already carry correct
   z_order values from SAM2Tracker.

3. Thing-on-thing same-Z conflict:
   If two thing instances share the same z_order in the same frame and
   their masks overlap, the lower-confidence instance gets z_order - 1.
   This is logged per frame. Confidence is stored in the instance dict
   from Stage 3 (SAM2Segmentor / Florence2Detector output).
   If no confidence is available, track_id is used as tiebreaker
   (higher track_id = lower priority).

4. Void is never generated. Leave those pixels uncovered.

5. Output track_id allocation:
   - Thing tracks: IDs from SAM2Tracker (already stable, start at 1+).
   - Stuff tracks: IDs allocated starting at STUFF_TRACK_ID_START (1000)
     to avoid collisions with thing track IDs.

Input contracts

stuff_maps : list[np.ndarray]
    One H×W uint8 project class_id map per frame (from SegFormerStuff).
    Pixel = 255 - unmapped/ignore.

thing_tracks : list[TrackAnnotation]
    Output of SAM2Tracker.finalize() - sparse keyframe list per instance.
    Passed through unchanged (Z-order already set, no modification).

instance_meta : dict[int, dict]
    track_id - {"label": str, "z_order": int, "confidence": float}
    Used only for same-Z conflict resolution. Built by pipeline.py from
    the Stage 3 / Stage 4 detection results.

config : dict
    Parsed auto_annotation.yaml (passed in, not re-read here).
    Used to look up z_order per label name.

Output contract

Returns tuple[list[TrackAnnotation], list[TrackAnnotation]]:
    (thing_tracks_resolved, stuff_tracks)
Both lists are ready for cvat_xml_exporter.py.
They are kept separate so the exporter can write stuff tracks first
(lower Z-order layers drawn first), then thing tracks on top.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from usv.auto_annotation.stuff.ade20k_mapping import ADE20K_TO_PROJECT

from usv.auto_annotation.types import PolygonKeyframe, TrackAnnotation
from usv.auto_annotation.postprocess.mask_utils import (
    mask_area,
    mask_iou,
    mask_to_polygon,
)

logger = logging.getLogger(__name__)

# Stuff tracks: keyframe threshold (IoU drop vs. previous keyframe)
STUFF_KEYFRAME_IOU_THRESH: float = 0.90

# Stuff track IDs start here to avoid collisions with thing track IDs
STUFF_TRACK_ID_START: int = 1000

# Minimum polygon area to emit a stuff polygon for a class on a frame
STUFF_MIN_AREA_PX: int = 64


# Public API

def resolve_overlaps(
    stuff_maps: list[np.ndarray],
    thing_tracks: list[TrackAnnotation],
    instance_meta: dict[int, dict],
    config: dict[str, Any],
) -> tuple[list[TrackAnnotation], list[TrackAnnotation]]:
    """
    Merge stuff maps and thing tracks into conflict-free TrackAnnotation lists.

    Parameters
    stuff_maps : list[np.ndarray]
        Per-frame H×W uint8 project class_id maps from SegFormerStuff.
        len(stuff_maps) == total clip frames.
    thing_tracks : list[TrackAnnotation]
        Sparse keyframe tracks from SAM2Tracker. Passed through with
        same-Z conflict resolution applied where needed.
    instance_meta : dict[int, dict]
        track_id - {"label", "z_order", "confidence"} for conflict resolution.
    config : dict
        Parsed auto_annotation.yaml.

    Returns
    tuple[list[TrackAnnotation], list[TrackAnnotation]]
        (resolved_thing_tracks, stuff_tracks)
    """
    # Build label - z_order lookup from config
    label_to_z: dict[str, int] = {
        lbl["name"]: lbl["z_order"] for lbl in config["labels"]
    }

    # Step 1: resolve thing-on-thing same-Z conflicts
    resolved_things = _resolve_thing_conflicts(
        thing_tracks, instance_meta, label_to_z
    )

    # Step 2: build stuff tracks from per-frame maps
    stuff_tracks = _build_stuff_tracks(stuff_maps, label_to_z, config)

    logger.info(
        "resolve_overlaps: %d thing tracks, %d stuff tracks produced.",
        len(resolved_things), len(stuff_tracks),
    )
    return resolved_things, stuff_tracks



# Thing-on-thing same-Z conflict resolution


def _resolve_thing_conflicts(
    thing_tracks: list[TrackAnnotation],
    instance_meta: dict[int, dict],
    label_to_z: dict[str, int],
) -> list[TrackAnnotation]:
    """
    Detect overlapping thing instances that share the same z_order in any
    frame, and demote the lower-priority one to z_order - 1.

    Priority: higher confidence wins. Tiebreak: lower track_id wins
    (earlier detection = more reliable).

    Operates on the polygon bounding boxes of keyframe polygons as an
    approximation - full pixel-level conflict detection is expensive on
    CPU and the polygon bbox overlap is sufficient for the draft annotation
    use case. Exact pixel conflicts are caught by overlap_conflict_test.py.

    Returns a new list of TrackAnnotation (modified copies where needed).
    """
    # Group tracks by z_order for fast conflict scanning
    z_to_tracks: dict[int, list[TrackAnnotation]] = {}
    for t in thing_tracks:
        z_to_tracks.setdefault(t.z_order, []).append(t)

    # Tracks that need z_order adjustment: track_id - new z_order
    z_adjustments: dict[int, int] = {}

    for z_order, tracks_at_z in z_to_tracks.items():
        if len(tracks_at_z) < 2:
            continue  # no conflict possible at this Z

        # Build frame_idx - list of (track_id, polygon_bbox) for this Z
        frame_index: dict[int, list[tuple[int, tuple[float,...]]]] = {}
        for track in tracks_at_z:
            for kf in track.keyframes:
                if kf.outside or not kf.points:
                    continue
                bbox = _polygon_bbox(kf.points)
                frame_index.setdefault(kf.frame_idx, []).append(
                    (track.track_id, bbox)
                )

        # Check each frame for bbox overlaps
        for frame_idx, entries in frame_index.items():
            if len(entries) < 2:
                continue
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    tid_a, bbox_a = entries[i]
                    tid_b, bbox_b = entries[j]
                    if not _bboxes_overlap(bbox_a, bbox_b):
                        continue
                    # Conflict: demote lower-priority track
                    loser = _pick_loser(tid_a, tid_b, instance_meta)
                    current_z = z_adjustments.get(loser, z_order)
                    z_adjustments[loser] = current_z - 1
                    logger.warning(
                        "resolve_overlaps: same-Z conflict (z=%d) "
                        "between track_id=%d and track_id=%d at frame %d "
                        "- track_id=%d demoted to z=%d",
                        z_order, tid_a, tid_b, frame_idx,
                        loser, z_adjustments[loser],
                    )

    # Apply adjustments - build new TrackAnnotation objects
    if not z_adjustments:
        return list(thing_tracks)

    result: list[TrackAnnotation] = []
    for track in thing_tracks:
        if track.track_id in z_adjustments:
            result.append(TrackAnnotation(
                track_id=track.track_id,
                label=track.label,
                z_order=z_adjustments[track.track_id],
                source=track.source,
                keyframes=track.keyframes,
            ))
        else:
            result.append(track)
    return result


def _pick_loser(
    tid_a: int,
    tid_b: int,
    instance_meta: dict[int, dict],
) -> int:
    """Return the track_id of the lower-priority instance."""
    conf_a = instance_meta.get(tid_a, {}).get("confidence", 0.0)
    conf_b = instance_meta.get(tid_b, {}).get("confidence", 0.0)
    if conf_a != conf_b:
        return tid_a if conf_a < conf_b else tid_b
    # Tiebreak: higher track_id = lower priority
    return max(tid_a, tid_b)


def _polygon_bbox(
    points: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Return (x1, y1, x2, y2) bounding box of a polygon."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _bboxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """Return True if two (x1,y1,x2,y2) bounding boxes intersect."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

# Build stuff tracks from per-frame class_id maps

def _build_stuff_tracks(
    stuff_maps: list[np.ndarray],
    label_to_z: dict[str, int],
    config: dict[str, Any],
) -> list[TrackAnnotation]:
    """
    Convert per-frame stuff class_id maps into TrackAnnotation objects.

    One TrackAnnotation is created per stuff class that appears in at
    least one frame. Keyframes are sparse: only frames where polygon IoU
    vs. the previous keyframe drops below STUFF_KEYFRAME_IOU_THRESH are
    included (plus always the first visible frame).

    Track IDs start at STUFF_TRACK_ID_START to avoid collisions.
    """
    # Build class_id - (label_name, z_order) from config
    mapped_class_ids = {cid for _, cid, _ in ADE20K_TO_PROJECT.values()}
    id_to_label = {
        lbl["id"]: (lbl["name"], lbl["z_order"])
        for lbl in config["labels"]
        if lbl.get("group") == "stuff" and lbl["id"] in mapped_class_ids
    }

    if not id_to_label:
        logger.warning(
            "_build_stuff_tracks: no stuff-group labels found in config. "
            "Check that config labels have group: stuff entries."
        )
        return []

    # Accumulate per-class per-frame binary masks
    # class_id - {frame_idx: binary_mask H×W uint8}
    class_frame_masks: dict[int, dict[int, np.ndarray]] = {
        cid: {} for cid in id_to_label
    }

    for frame_idx, proj_map in enumerate(stuff_maps):
        for class_id in id_to_label:
            binary = (proj_map == class_id).astype(np.uint8)
            if mask_area(binary) >= STUFF_MIN_AREA_PX:
                class_frame_masks[class_id][frame_idx] = binary

    stuff_tracks: list[TrackAnnotation] = []
    track_id_counter = STUFF_TRACK_ID_START

    for class_id, frame_mask_map in class_frame_masks.items():
        if not frame_mask_map:
            logger.debug(
                "_build_stuff_tracks: class_id=%d has no frames "
                "with area >= %d px² - skipped.",
                class_id, STUFF_MIN_AREA_PX,
            )
            continue

        label_name, z_order = id_to_label[class_id]
        keyframes = _stuff_keyframes(
            frame_mask_map, label_name, class_id
        )
        if not keyframes:
            continue

        stuff_tracks.append(TrackAnnotation(
            track_id=track_id_counter,
            label=label_name,
            z_order=z_order,
            source="auto",
            keyframes=keyframes,
        ))
        logger.info(
            "_build_stuff_tracks: class '%s' (id=%d, z=%d) - "
            "track_id=%d with %d keyframes across %d frames.",
            label_name, class_id, z_order,
            track_id_counter, len(keyframes), len(frame_mask_map),
        )
        track_id_counter += 1

    return stuff_tracks


def _stuff_keyframes(
    frame_mask_map: dict[int, np.ndarray],
    label_name: str,
    class_id: int,
) -> list[PolygonKeyframe]:
    """
    Build sparse keyframe list for one stuff class.

    Keyframe selection
    ------------------
    - First frame with this class: always keyframe=True.
    - Subsequent frames: keyframe=True if IoU(current_mask, last_kf_mask)
      < STUFF_KEYFRAME_IOU_THRESH (0.90).
    - Non-keyframe frames are skipped (CVAT interpolates between keyframes).
    - If mask_to_polygon returns < 3 points the frame is skipped with a
      warning (degenerate contour - too small for a valid polygon).
    """
    sorted_frames = sorted(frame_mask_map.keys())
    keyframe_list: list[PolygonKeyframe] = []
    last_kf_mask: np.ndarray | None = None

    for frame_idx in sorted_frames:
        mask = frame_mask_map[frame_idx]
        is_first = (last_kf_mask is None)

        if is_first:
            is_keyframe = True
        else:
            iou = mask_iou(mask, last_kf_mask)
            is_keyframe = iou < STUFF_KEYFRAME_IOU_THRESH

        if not is_keyframe:
            continue

        polygon = mask_to_polygon(mask)
        if len(polygon) < 3:
            logger.debug(
                "_stuff_keyframes: class '%s' frame %d - "
                "degenerate polygon (%d pts) skipped.",
                label_name, frame_idx, len(polygon),
            )
            # Do not update last_kf_mask - wait for a valid polygon
            continue

        keyframe_list.append(PolygonKeyframe(
            frame_idx=frame_idx,
            points=polygon,
            keyframe=True,
            outside=False,
            occluded=False,
        ))
        last_kf_mask = mask  # update reference

    return keyframe_list
