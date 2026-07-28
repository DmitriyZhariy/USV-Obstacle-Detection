"""
Generates fixed-length non-overlapping video clips centered around keyframes.

For each keyframe, a clip of [kf - context_sec, kf + context_sec] is created.
If a keyframe falls inside a previously created clip's window, it is skipped.
If clips would overlap, the later clip's start is trimmed to the previous clip's end.
"""
import argparse
import re
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class SequenceDatasetBuilder:
    def __init__(
        self,
        raw_dirs: list[str],
        squeezed_dir: str,
        output_dir: str,
        context_sec: float = 2.5,
        target_fps: int = 5,
        original_interval_sec: float = 5.0,
        skip_existing: bool = True,
    ):
        self.raw_dirs = [Path(d) for d in raw_dirs]
        self.squeezed_dir = Path(squeezed_dir)
        self.output_dir = Path(output_dir)

        self.context_sec = context_sec
        self.target_fps = target_fps
        self.original_interval_sec = original_interval_sec
        self.skip_existing = skip_existing

        self.videos_dir = self.output_dir / "videos"
        self.frames_dir = self.output_dir / "frames"
        self.metadata_dir = self.output_dir / "metadata"

        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.mapping: dict = {}
    
    def build_index_mapping(self):
        """
        Takes the selected frames and restores their original source
        to build an exact mapping map: global_idx -> (video_path, frame_idx, prefix)
        """
        video_files = []
        for p in self.raw_dirs:
            for ext in ["*.avi", "*.mp4", "*.MOV", "*.mkv"]:
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

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            extraction_step = max(1, int(fps * self.original_interval_sec))

            for i in range(0, frame_count, extraction_step):
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

    def _parse_keyframes_by_video(self) -> dict:
        pattern = re.compile(r".*_(\d+)\.(jpg|jpeg|png)$")
        result: dict = {}

        for file_path in self.squeezed_dir.glob("*.*"):
            match = pattern.match(file_path.name)
            if not match:
                continue

            g_idx = int(match.group(1))
            if g_idx not in self.mapping:
                continue

            meta = self.mapping[g_idx]
            v_path = meta["video_path"]

            if v_path not in result:
                result[v_path] = {
                    "prefix": meta["prefix"],
                    "fps": meta["fps"],
                    "keyframes": [],
                }
            result[v_path]["keyframes"].append(meta["frame_idx"])

        for v_path in result:
            result[v_path]["keyframes"].sort()

        return result

    def _build_non_overlapping_clips(
        self, keyframes: list[int], fps: float
    ) -> list[tuple]:
        clips = []
        last_end_frame = -1

        for kf_frame in keyframes:
            context_frames = int(self.context_sec * fps)

            if kf_frame <= last_end_frame:
                continue

            actual_start = max(kf_frame - context_frames, last_end_frame + 1)
            actual_end = kf_frame + context_frames

            clips.append((actual_start, actual_end))
            last_end_frame = actual_end

        return clips

    def extract_sequences(self):
        video_data = self._parse_keyframes_by_video()
        print(f"Found keyframes in {len(video_data)} videos.")

        for v_path, data in tqdm(
            video_data.items(), desc="Extracting clips"
        ):
            prefix = data["prefix"]
            fps = data["fps"]
            clips = self._build_non_overlapping_clips(data["keyframes"], fps)
            if not clips:
                continue

            cap = cv2.VideoCapture(str(v_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_stem = v_path.stem

            for seq_id, (start, end) in enumerate(clips):
                # Clamp end frame to actual video length
                end = min(end, total_frames - 1)
                
                # Naming convention: right_MOVI0017_0001
                clip_name = f"{prefix}_{video_stem}_{seq_id + 1:04d}"

                mp4_path = self.videos_dir / f"{clip_name}.mp4"
                csv_path = self.metadata_dir / f"{clip_name}.csv"

                # Idempotency: skip already generated clips
                if self.skip_existing and mp4_path.exists():
                    print(f"Skipping existing clip: {clip_name}")
                    continue

                # Define specific paths for this clip
                clip_frames_dir = self.frames_dir / clip_name
                clip_frames_dir.mkdir(parents=True, exist_ok=True)

                frame_step = max(1, int(fps / self.target_fps))
                metadata_rows = []
                writer = None
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                frame_counter = 1

                for i in range(start, end + 1):
                    if (i - start) % frame_step != 0:
                        continue

                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret: break

                    if prefix == "center":
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    if writer is None:
                        h, w, _ = frame.shape
                        writer = cv2.VideoWriter(
                            str(mp4_path), fourcc, self.target_fps, (w, h)
                        )

                    writer.write(frame)

                    # Save JPEG sequence: 0001.jpeg, 0002.jpeg
                    img_name = f"{frame_counter:04d}.jpeg"
                    cv2.imwrite(str(clip_frames_dir / img_name), frame)
                    metadata_rows.append(
                        {
                            "local_seq_idx": frame_counter,
                            "original_video": v_path.name,
                            "original_frame_idx": i,
                        }
                    )
                    frame_counter += 1

                if writer:
                    writer.release()

                # Save CSV explicitly into the metadata directory
                pd.DataFrame(metadata_rows).to_csv(csv_path, index=False)

            cap.release()
        print(f"All sequences extracted to {self.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fixed-length non-overlapping clips from keyframes."
    )
    parser.add_argument(
        "--raw-dirs", nargs="+",
        default=["data/raw/left_cam", "data/raw/right_cam", "data/raw/center_phone"],
        help="Paths to raw video directories.",
    )
    parser.add_argument(
        "--squeezed-dir",
        default="data/interim/labeling_v5_mf_squeezed",
        help="Directory with selected keyframe images.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim/sequent_frames_v5",
        help="Output directory for clips.",
    )
    parser.add_argument(
        "--context-sec", type=float, default=2.5,
        help="Half-window around each keyframe in seconds (default: 2.5).",
    )
    parser.add_argument(
        "--target-fps", type=int, default=5,
        help="Target FPS for output clips (default: 5).",
    )
    parser.add_argument(
        "--original-interval-sec", type=float, default=5.0,
        help="Sampling interval used during frame extraction (default: 5.0).",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Regenerate clips even if they already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    builder = SequenceDatasetBuilder(
        raw_dirs=args.raw_dirs,
        squeezed_dir=args.squeezed_dir,
        output_dir=args.output_dir,
        context_sec=args.context_sec,
        target_fps=args.target_fps,
        original_interval_sec=args.original_interval_sec,
        skip_existing=not args.no_skip_existing,
    )

    builder.build_index_mapping()
    builder.extract_sequences()