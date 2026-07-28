"""
Extracts frames from raw video files at a fixed time interval.

Supports multi-directory input and three camera prefixes derived from the
directory path: left, right, center. Center (phone) frames are optionally
rotated 90° clockwise to correct portrait-mode recording.

Output naming: {prefix}_{global_idx:05d}.jpg
Global index is a single counter across all videos from all input directories.
"""
import argparse
import cv2
from pathlib import Path
import tqdm


def extract_mixed_frames(
    input_dirs: list[str],
    output_dir: str,
    interval_sec: int = 5,
    rotate_center: bool = True,
) -> None:
    """Extract one frame every `interval_sec` seconds from all videos in `input_dirs`.

    Camera prefix is inferred from the video file path:
    - "phone" or "center" in path → prefix "center"
    - "right" in path             → prefix "right"
    - otherwise                   → prefix "left"

    Args:
        input_dirs: List of directories containing raw video files.
        output_dir: Destination directory for extracted JPEG frames.
        interval_sec: Sampling interval in seconds between extracted frames.
        rotate_center: If True, rotate center-camera frames 90° clockwise.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_files: list[Path] = []
    for d in input_dirs:
        p = Path(d)
        for ext in ["*.avi", "*.mp4", "*.MOV", "*.mkv"]:
            video_files.extend(list(p.rglob(ext)))

    video_files.sort()
    print(f"Found {len(video_files)} videos in total.")

    global_idx = 0
    failed_videos: list[str] = []
    failed_frames: list[str] = []

    for vid_file in tqdm.tqdm(video_files, desc="Extracting"):
        cap = cv2.VideoCapture(str(vid_file))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {vid_file}")
            failed_videos.append(str(vid_file))
            continue

        path_str = str(vid_file).lower()
        if "phone" in path_str or "center" in path_str:
            prefix = "center"
        elif "right" in path_str:
            prefix = "right"
        else:
            prefix = "left"

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(fps * interval_sec))

        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                break

            if prefix == "center" and rotate_center:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            fname = f"{prefix}_{global_idx:05d}.jpg"
            cv2.imwrite(str(output_path / fname), frame)
            global_idx += 1

        cap.release()

    print(f"\nDone. Saved {global_idx} frames to '{output_dir}'.")

    if failed_videos:
        print(f"[WARNING] {len(failed_videos)} video(s) could not be opened:")
        for v in failed_videos:
            print(f"  - {v}")
    if failed_frames:
        print(f"[WARNING] {len(failed_frames)} frame(s) could not be read.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from raw video files at a fixed time interval."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=["data/raw/left_cam", "data/raw/right_cam", "data/raw/center_phone"],
        help="Paths to raw video directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim/labeling_v10",
        help="Directory to save extracted JPEG frames.",
    )
    parser.add_argument(
        "--interval-sec",
        type=int,
        default=5,
        help="Sampling interval in seconds between extracted frames (default: 5).",
    )
    parser.add_argument(
        "--no-rotate-center",
        action="store_false",
        dest="rotate_center",
        help="Disable 90° clockwise rotation for center-camera frames.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_mixed_frames(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        interval_sec=args.interval_sec,
        rotate_center=args.rotate_center,
    )
