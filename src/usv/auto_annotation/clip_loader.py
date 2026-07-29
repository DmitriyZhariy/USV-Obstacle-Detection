"""
Loads a single clip (JPEG sequence + CSV metadata) from sequent_frames output.

Frame files: {clips_dir}/frames/{clip_name}/{local_seq_idx:04d}.jpeg  (1-based)
Metadata:    {clips_dir}/metadata/{clip_name}.csv
Keyframe:    derived as len(frames) // 2  (center frame = original keyframe)
"""
import logging
import cv2
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from usv.auto_annotation.types import ClipData

logger = logging.getLogger(__name__)


class ClipNotFoundError(FileNotFoundError):
    """Raised when the clip frames directory or metadata CSV does not exist."""


class ClipLoader:
    def __init__(self, clips_dir: Path):
        """
        Args:
            clips_dir: Root output dir from generate_sequences.py
                       (contains subfolders: frames/, metadata/, videos/)
        """
        self.clips_dir = Path(clips_dir)
        self.frames_root = self.clips_dir / "frames"
        self.metadata_root = self.clips_dir / "metadata"

    def load(self, clip_name: str) -> ClipData:
        """Load all frames and metadata for a clip.

        Args:
            clip_name: Clip identifier, e.g. "right_MOVI0023_0004"

        Returns:
            ClipData with frames list (BGR), derived keyframe_idx, and meta_df.

        Raises:
            ClipNotFoundError: frames dir or CSV is missing.
        """
        frames_dir = self.frames_root / clip_name
        csv_path = self.metadata_root / f"{clip_name}.csv"

        if not frames_dir.exists():
            raise ClipNotFoundError(f"Frames directory not found: {frames_dir}")
        if not csv_path.exists():
            raise ClipNotFoundError(f"Metadata CSV not found: {csv_path}")

        meta_df = pd.read_csv(csv_path)
        expected_n = len(meta_df)

        # Load frames in local_seq_idx order (1-based → 0001.jpeg ... NNNN.jpeg)
        frames: list = []
        for idx in meta_df["local_seq_idx"]:
            img_path = frames_dir / f"{idx:04d}.jpeg"
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning("Could not read frame: %s — skipping", img_path)
                continue
            frames.append(frame)

        actual_n = len(frames)
        if actual_n < expected_n:
            logger.warning(
                "Clip %s: expected %d frames, loaded %d. Some JPEG files may be missing.",
                clip_name, expected_n, actual_n,
            )

        h, w = frames[0].shape[:2] if frames else (0, 0)
        keyframe_idx = actual_n // 2

        original_video = meta_df["original_video"].iloc[0] if not meta_df.empty else ""

        return ClipData(
            clip_name=clip_name,
            frames=frames,
            keyframe_idx=keyframe_idx,
            n_frames=actual_n,
            frame_height=h,
            frame_width=w,
            meta_df=meta_df,
            original_video=original_video,
        )
    
    def load_from_dir(self, frames_dir: Path) -> ClipData:
        """
        Load a clip directly from a frames directory (no metadata CSV required).

        Frame files: {frames_dir}/{local_seq_idx:04d}.jpeg  (1-based)
        clip_name is derived from frames_dir.name.
        keyframe_idx is derived as len(frames) // 2.
        """
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            raise ClipNotFoundError(f"Frames directory not found: {frames_dir}")

        clip_name = frames_dir.name

        jpeg_files = sorted(frames_dir.glob("*.jpeg"))
        if not jpeg_files:
            raise ClipNotFoundError(f"No .jpeg files found in: {frames_dir}")

        frames = []
        seq_indices = []
        for img_path in jpeg_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning("Could not read frame: %s — skipping", img_path)
                continue
            frames.append(frame)
            try:
                seq_indices.append(int(img_path.stem))
            except ValueError:
                seq_indices.append(len(seq_indices) + 1)

        if not frames:
            raise ClipNotFoundError(f"No readable frames in: {frames_dir}")

        actual_n = len(frames)
        h, w = frames[0].shape[:2]
        keyframe_idx = actual_n // 2

        meta_df = pd.DataFrame({
            "local_seq_idx": seq_indices,
            "original_video": [clip_name] * actual_n,
            "original_frame_idx": seq_indices,
        })

        return ClipData(
            clip_name=clip_name,
            frames=frames,
            keyframe_idx=keyframe_idx,
            n_frames=actual_n,
            frame_height=h,
            frame_width=w,
            meta_df=meta_df,
            original_video=clip_name,
        )

    def list_clips(self) -> list[str]:
        """Return sorted list of all available clip names."""
        if not self.frames_root.exists():
            return []
        return sorted(p.name for p in self.frames_root.iterdir() if p.is_dir())