"""
Script to restore metadata from squeezed frames and generate dense sequential clips.
Merges overlapping context intervals, applies necessary rotations, and saves
outputs directly into a structured dataset format (videos/, frames/, metadata/).
"""
import os
import cv2
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class SequenceDatasetBuilder:
    def __init__(self, raw_dirs, squeezed_dir, output_dir, context_sec=5, target_fps=5):
        self.raw_dirs = [Path(d) for d in raw_dirs]
        self.squeezed_dir = Path(squeezed_dir)
        self.output_dir = Path(output_dir)

        self.context_sec = context_sec
        self.target_fps = target_fps

        # Define exact output subdirectories
        self.videos_dir = self.output_dir / "videos"
        self.frames_dir = self.output_dir / "frames"
        self.metadata_dir = self.output_dir / "metadata"

        # Create all necessary directories
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.mapping = {}

    def build_index_mapping(self, original_interval_sec=5):
        """
        Takes the selected frames and restores their original source
        to build an exact mapping map: global_idx -> (video_path, frame_idx, prefix)
        """
        video_files =[]
        for p in self.raw_dirs:
            for ext in['*.avi', '*.mp4', '*.MOV', '*.mkv']:
                video_files.extend(list(p.rglob(ext)))

        video_files.sort()
        print(f"Building mapping for {len(video_files)} videos...")

        global_idx = 0

        for vid_file in tqdm(video_files, desc="Simulating extraction"):
            cap = cv2.VideoCapture(str(vid_file))
            if not cap.isOpened():
                continue

            path_str = str(vid_file).lower()
            if "phone" in path_str or "center" in path_str:
                prefix = "center"
            elif "right" in path_str:
                prefix = "right"
            else:
                prefix = "left"

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = int(fps * original_interval_sec)
            if step == 0: step = 1

            for i in range(0, frame_count, step):
                # Fast check if the frame exists
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                if not cap.grab():
                    break

                self.mapping[global_idx] = {
                    'video_path': vid_file,
                    'frame_idx': i,
                    'prefix': prefix,
                    'fps': fps
                }
                global_idx += 1

            cap.release()
        print(f"Mapping built. Total anchors matched: {len(self.mapping)}")

    def parse_squeezed_frames(self):
        """
        Parses the folder with squeezed frames and extracts global indices.
        Returns intervals grouped by video files.
        """
        intervals_by_video = {}

        # Regex to extract numbers from format prefix_00123.jpg
        pattern = re.compile(r".*_(\d+)\.(jpg|jpeg|png)$")

        for file_path in self.squeezed_dir.glob("*.*"):
            match = pattern.match(file_path.name)
            if match:
                g_idx = int(match.group(1))
                if g_idx in self.mapping:
                    meta = self.mapping[g_idx]
                    v_path = meta['video_path']
                    f_idx = meta['frame_idx']
                    fps = meta['fps']

                    # Calculate context in original frames based on video FPS
                    frames_context = int(self.context_sec * fps)
                    start_f = max(0, f_idx - frames_context)
                    end_f = f_idx + frames_context

                    if v_path not in intervals_by_video:
                        intervals_by_video[v_path] =[]

                    intervals_by_video[v_path].append([start_f, end_f, meta['prefix'], fps])
                else:
                    print(f"Warning: Frame {file_path.name} not found in mapping.")

        return intervals_by_video

    def merge_intervals(self, intervals):
        """Merges overlapping or adjacent intervals for a single video."""
        if not intervals: return[]
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]

        for current in intervals[1:]:
            prev = merged[-1]
            # If intervals overlap or are adjacent
            if current[0] <= prev[1]:
                prev[1] = max(prev[1], current[1])
            else:
                merged.append(current)
        return merged

    def extract_sequences(self):
        """Extracts frames, applies rotation, and saves final datasets in structured folders."""
        intervals_by_video = self.parse_squeezed_frames()
        print(f"Found anchors in {len(intervals_by_video)} videos.")

        for v_path, raw_intervals in tqdm(intervals_by_video.items(), desc="Extracting clips"):
            merged_intervals = self.merge_intervals(raw_intervals)

            cap = cv2.VideoCapture(str(v_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Extract metadata from the first interval
            prefix = merged_intervals[0][2]
            fps = merged_intervals[0][3]
            frame_step = int(max(1, fps / self.target_fps))

            video_stem = v_path.stem

            for seq_id, (start, end, _, _) in enumerate(merged_intervals):
                # Clamp end frame to actual video length
                end = min(end, total_frames - 1)

                # Naming convention: right_MOVI0017_0001
                clip_name = f"{prefix}_{video_stem}_{seq_id + 1:04d}"

                # Define specific paths for this clip
                clip_frames_dir = self.frames_dir / clip_name
                clip_frames_dir.mkdir(parents=True, exist_ok=True)

                mp4_path = self.videos_dir / f"{clip_name}.mp4"
                csv_path = self.metadata_dir / f"{clip_name}.csv"

                metadata_rows =[]
                writer = None
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                frame_counter = 1

                for i in range(start, end + 1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret: break

                    # Sample frames only at the desired step (e.g., every 6th for 5 FPS)
                    if (i - start) % frame_step == 0:

                        # Apply rotation fix for center phone camera
                        if prefix == 'center':
                            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                        # Initialize VideoWriter upon receiving the first valid frame
                        if writer is None:
                            h, w, _ = frame.shape
                            writer = cv2.VideoWriter(str(mp4_path), fourcc, self.target_fps, (w, h))

                        writer.write(frame)

                        # Save JPEG sequence: 0001.jpeg, 0002.jpeg
                        img_name = f"{frame_counter:04d}.jpeg"
                        cv2.imwrite(str(clip_frames_dir / img_name), frame)

                        metadata_rows.append({
                            'local_seq_idx': frame_counter,
                            'original_video': v_path.name,
                            'original_frame_idx': i
                        })
                        frame_counter += 1

                if writer:
                    writer.release()

                # Save CSV explicitly into the metadata directory
                pd.DataFrame(metadata_rows).to_csv(csv_path, index=False)

            cap.release()
        print(f"All sequences extracted to {self.output_dir}")

if __name__ == "__main__":
    raw_folders =[
        "data/raw/left_cam",
        "data/raw/right_cam",
        "data/raw/center_phone"
    ]

    # Initialize builder
    builder = SequenceDatasetBuilder(
        raw_dirs=raw_folders,
        squeezed_dir="data/interim/labeling_v5_mf_squeezed",
        output_dir="data/interim/TEST_sequent_frames_v5",
        context_sec=5,  # 5 seconds before and 5 after the keyframe
        target_fps=5    # Target FPS of the output clip
    )

    # 1. Restore metadata mapping
    builder.build_index_mapping(original_interval_sec=5)

    # 2. Extract and format sequences
    builder.extract_sequences()
