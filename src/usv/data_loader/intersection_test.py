"""
Tests that consecutive clips from the same source video do not overlap.

For each pair of consecutive clips (by seq_id) sharing the same original_video,
the first original_frame_idx of clip N+1 must be strictly greater than
the last original_frame_idx of clip N.
"""
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CLIP_NAME_RE = re.compile(r"^(?P<prefix>left|right|center)_(?P<rest>.+)_(?P<seq_id>\d{4})$")


@dataclass
class ClipInfo:
    clip_name: str
    prefix: str
    video_stem: str
    seq_id: int
    first_frame_idx: int
    last_frame_idx: int


def load_clips(frames_dir: Path, metadata_dir: Path) -> list[ClipInfo]:
    clips = []
    missing_csv = []

    for clip_folder in sorted(frames_dir.iterdir()):
        if not clip_folder.is_dir():
            continue

        m = CLIP_NAME_RE.match(clip_folder.name)
        if not m:
            print(f"  [SKIP] Cannot parse clip name: {clip_folder.name}")
            continue

        prefix = m.group("prefix")
        seq_id = int(m.group("seq_id"))

        csv_path = metadata_dir / f"{clip_folder.name}.csv"
        if not csv_path.exists():
            missing_csv.append(clip_folder.name)
            continue

        df = pd.read_csv(csv_path)
        if df.empty or "original_frame_idx" not in df.columns:
            print(f"  [SKIP] Empty or malformed CSV: {csv_path.name}")
            continue

        video_stem = Path(df["original_video"].iloc[0]).stem
        frame_idxs = df["original_frame_idx"].sort_values().tolist()

        clips.append(ClipInfo(
            clip_name=clip_folder.name,
            prefix=prefix,
            video_stem=video_stem,
            seq_id=seq_id,
            first_frame_idx=int(frame_idxs[0]),
            last_frame_idx=int(frame_idxs[-1]),
        ))

    if missing_csv:
        print(f"\n[WARNING] {len(missing_csv)} clip(s) have no metadata CSV and were skipped:")
        for name in missing_csv:
            print(f"  - {name}")

    return clips


def group_by_video(clips: list[ClipInfo]) -> dict[str, list[ClipInfo]]:
    groups: dict[str, list[ClipInfo]] = {}
    for clip in clips:
        key = f"{clip.prefix}/{clip.video_stem}"
        groups.setdefault(key, []).append(clip)
    for key in groups:
        groups[key].sort(key=lambda c: c.seq_id)
    return groups


def check_intersections(groups: dict[str, list[ClipInfo]]) -> list[dict]:
    violations = []
    for video_stem, clips in sorted(groups.items()):
        for i in range(len(clips) - 1):
            a, b = clips[i], clips[i + 1]
            if b.first_frame_idx <= a.last_frame_idx:
                violations.append({
                    "video": video_stem,
                    "clip_a": a.clip_name,
                    "clip_b": b.clip_name,
                    "last_frame_a": a.last_frame_idx,
                    "first_frame_b": b.first_frame_idx,
                    "overlap_frames": a.last_frame_idx - b.first_frame_idx + 1,
                })
    return violations


def report(groups: dict[str, list[ClipInfo]], violations: list[dict]):
    violation_videos = {v["video"] for v in violations}

    print("INTERSECTION TEST RESULTS")

    for video_stem, clips in sorted(groups.items()):
        status = "FAIL" if video_stem in violation_videos else "OK"
        print(f"  [{status}]  {video_stem}  ({len(clips)} clips)")

    if violations:
        print(f"\n{len(violations)} VIOLATION(S) DETECTED:\n")
        for v in violations:
            print(
                f"  {v['clip_a']}  →  {v['clip_b']}\n"
                f"    last frame of A : {v['last_frame_a']}\n"
                f"    first frame of B: {v['first_frame_b']}\n"
                f"    overlap         : {v['overlap_frames']} frame(s)\n"
            )
        print("RESULT: FAIL")
    else:
        print("RESULT: PASS — no intersections found")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test that consecutive clips from the same video do not overlap."
    )
    parser.add_argument(
        "--frames-dir",
        default="data/interim/sequent_frames_v5-1/frames",
        help="Path to the frames/ directory with per-clip subfolders.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="data/interim/sequent_frames_v5-1/metadata",
        help="Path to the metadata/ directory with per-clip CSVs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    metadata_dir = Path(args.metadata_dir)

    print(f"Loading clips from : {frames_dir}")
    print(f"Reading metadata from: {metadata_dir}")

    clips = load_clips(frames_dir, metadata_dir)
    print(f"Loaded {len(clips)} clip(s).")

    groups = group_by_video(clips)
    print(f"Grouped into {len(groups)} source video(s).")

    violations = check_intersections(groups)
    report(groups, violations)

    sys.exit(1 if violations else 0)
