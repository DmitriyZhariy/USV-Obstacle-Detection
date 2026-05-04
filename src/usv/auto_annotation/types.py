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