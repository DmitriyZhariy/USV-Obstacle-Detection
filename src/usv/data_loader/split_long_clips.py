"""
Splits overly long generated sequences into smaller chunks (e.g., max 30 seconds).
Preserves the base ID naming convention and appends '_partX' to ensure
consistency with manual documentation and future pipeline integration.
"""
import os
import cv2
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

class SequenceSplitter:
    def __init__(self, dataset_dir, max_seconds=30, target_fps=5):
        self.dataset_dir = Path(dataset_dir)
        self.videos_dir = self.dataset_dir / "videos"
        self.frames_dir = self.dataset_dir / "frames"
        self.metadata_dir = self.dataset_dir / "metadata"

        self.max_frames = max_seconds * target_fps
        self.target_fps = target_fps

    def split_all(self):
        # Find all metadata CSVs to determine which clips exist
        csv_files = list(self.metadata_dir.glob("*.csv"))

        clips_to_split =[]
        for csv_path in csv_files:
            # Avoid splitting already split clips if script is run twice
            if "_part" in csv_path.stem:
                continue

            df = pd.read_csv(csv_path)
            if len(df) > self.max_frames:
                clips_to_split.append((csv_path.stem, df))

        print(f"Found {len(clips_to_split)} clips longer than {self.max_frames} frames.")

        for clip_name, df in tqdm(clips_to_split, desc="Splitting clips"):
            self._process_clip(clip_name, df)

        print("Splitting complete. Cleaned up original oversized files.")

    def _process_clip(self, clip_name, df):
        total_frames = len(df)
        num_parts = (total_frames + self.max_frames - 1) // self.max_frames

        orig_frames_dir = self.frames_dir / clip_name
        orig_video_path = self.videos_dir / f"{clip_name}.mp4"
        orig_csv_path = self.metadata_dir / f"{clip_name}.csv"

        for part_idx in range(num_parts):
            part_name = f"{clip_name}_part{part_idx + 1}"

            # Sub-dataframe for the chunk
            start_row = part_idx * self.max_frames
            end_row = min((part_idx + 1) * self.max_frames, total_frames)
            df_part = df.iloc[start_row:end_row].copy()

            # Create new directories and paths
            new_frames_dir = self.frames_dir / part_name
            new_frames_dir.mkdir(parents=True, exist_ok=True)

            new_csv_path = self.metadata_dir / f"{part_name}.csv"
            new_video_path = self.videos_dir / f"{part_name}.mp4"

            # 1. Move and rename JPEGs, update CSV
            writer = None
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Reset local_seq_idx for the chunk to start from 1
            df_part['local_seq_idx'] = range(1, len(df_part) + 1)

            for index, row in df_part.iterrows():
                # Original JPEG name (e.g., 0151.jpeg)
                orig_idx = index + 1
                orig_img_path = orig_frames_dir / f"{orig_idx:04d}.jpeg"

                # New JPEG name (e.g., 0001.jpeg)
                new_idx = row['local_seq_idx']
                new_img_path = new_frames_dir / f"{new_idx:04d}.jpeg"

                # Copy image instead of moving right away (safety first)
                if orig_img_path.exists():
                    shutil.copy2(orig_img_path, new_img_path)

                    # 2. Build the new MP4 video from the JPEGs
                    frame_img = cv2.imread(str(new_img_path))
                    if writer is None:
                        h, w, _ = frame_img.shape
                        writer = cv2.VideoWriter(str(new_video_path), fourcc, self.target_fps, (w, h))
                    writer.write(frame_img)

            if writer:
                writer.release()

            # 3. Save the updated CSV
            df_part.to_csv(new_csv_path, index=False)

        # 4. Cleanup original files ONLY after successful splitting
        shutil.rmtree(orig_frames_dir)
        if orig_video_path.exists():
            orig_video_path.unlink()
        if orig_csv_path.exists():
            orig_csv_path.unlink()

if __name__ == "__main__":
    dataset_path = "data/interim/sequent_frames_v5"

    # max_seconds=30 at target_fps=5 means chunks of 150 frames max
    splitter = SequenceSplitter(
        dataset_dir=dataset_path,
        max_seconds=30,
        target_fps=5
    )

    splitter.split_all()
