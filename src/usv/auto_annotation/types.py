"""Shared dataclasses for the auto_annotation pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class ClipData:
    clip_name: str # "right_MOVI0023_0004"
    frames: list[np.ndarray] # BGR uint8, ordered, 1-indexed position = local_seq_idx - 1
    keyframe_idx: int # 0-based index into frames[], derived as len(frames) // 2
    n_frames: int
    frame_height: int
    frame_width: int
    meta_df: pd.DataFrame # columns: local_seq_idx, original_video, original_frame_idx
    original_video: str # "MOVI0023.avi"  (from meta_df, unique value)


@dataclass
class PolygonKeyframe:
    """A single polygon annotation at one frame index inside a track."""
    frame_idx: int # 0-based frame index in the clip
    points: list[tuple[float, float]] # [(x0,y0), (x1,y1), ...] in original resolution
    keyframe: bool = True # True - CVAT renders this frame; False - interpolated
    outside: bool = False # True - object left the frame at this point
    occluded: bool = False # always False from auto-annotation


@dataclass
class TrackAnnotation:
    """Full track for one object instance across the clip."""
    track_id: int
    label: str
    z_order: int # fixed per annotation manual table
    source: str = "auto" # marks auto-generated tracks
    keyframes: list[PolygonKeyframe] = field(default_factory=list)
