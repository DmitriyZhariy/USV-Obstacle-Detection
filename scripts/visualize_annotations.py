"""
Render CVAT 1.1 track-format annotations onto a video clip.

Reads the annotations.xml produced by run_auto_annotation.py (cpu-sam2 mode),
interpolates polygons between sparse keyframes, and writes either:
  - a debug JPEG strip  (default, --output-frames)
  - an MP4 video        (--output-video)
  - both                (both flags together)

Usage – single clip (paths resolved from project defaults):
    uv run python -m scripts.visualize_annotations \
        --clip-name right_MOVI0017_0001

Usage – explicit paths + video output:
    uv run python -m scripts.visualize_annotations \
        --clip-name right_MOVI0017_0001 \
        --output-video out/right_MOVI0017_0001.mp4 \
        --fps 5 \
        --opacity 0.35

Notes
-----
- Handles <track> elements with sparse polygon keyframes + linear interpolation.
- Polygons marked outside="1" are skipped (object not visible).
- z_order is respected: lower z_order layers drawn first (underneath).
- Clip frames are read from <clips-dir>/frames/<clip-name>/ as JPEG/PNG.
- Requires only opencv-python and numpy (already in project deps).
"""
from __future__ import annotations

import argparse
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label palette (BGR) – matches existing cvat_polygons_to_mp4.py colours
# ---------------------------------------------------------------------------
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "Sky":          (135, 206, 235),
    "Water":        ( 30, 144, 255),
    "Land":         ( 34, 139,  34),
    "Bridge":       (112, 128, 144),
    "Pier":         (139,  69,  19),
    "Vessel":       (255,  69,   0),
    "Buoy":         (255, 215,   0),
    "LandingMark":  (255,   0, 255),
    "BridgeLight":  (  0, 255,   0),
    "Other":        (255, 255, 255),
    "Void":         (  0,   0,   0),
}
_DEFAULT_COLOR = (128, 128, 128)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class _Keyframe(NamedTuple):
    frame_idx: int
    points: np.ndarray      # shape (N, 2), float32, pixel coords
    outside: bool


class _Track(NamedTuple):
    label: str
    z_order: int
    keyframes: list[_Keyframe]


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _parse_points(points_str: str) -> np.ndarray:
    pts = []
    for pair in points_str.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        pts.append([float(x_str), float(y_str)])
    return np.array(pts, dtype=np.float32)


def parse_tracks(xml_path: Path) -> list[_Track]:
    """Parse all <track> elements from a CVAT 1.1 XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tracks: list[_Track] = []
    for track_el in root.findall(".//track"):
        label    = track_el.get("label", "")
        z_order  = int(track_el.get("z_order", "0"))
        kfs: list[_Keyframe] = []

        for poly_el in track_el.findall("polygon"):
            frame_idx = int(poly_el.get("frame", "0"))
            outside   = poly_el.get("outside", "0") == "1"
            points    = _parse_points(poly_el.get("points", ""))
            kfs.append(_Keyframe(frame_idx=frame_idx, points=points, outside=outside))

        kfs.sort(key=lambda k: k.frame_idx)
        if kfs:
            tracks.append(_Track(label=label, z_order=z_order, keyframes=kfs))

    logger.info("Parsed %d tracks from %s", len(tracks), xml_path.name)
    return tracks


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def _lerp_polygon(pts_a: np.ndarray, pts_b: np.ndarray, t: float) -> np.ndarray:
    """
    Linear interpolation between two polygons.
    Falls back to nearest keyframe when vertex counts differ.
    """
    if pts_a.shape == pts_b.shape:
        return (1.0 - t) * pts_a + t * pts_b
    return pts_a if t < 0.5 else pts_b


def get_polygon_at_frame(track: _Track, frame_idx: int) -> np.ndarray | None:
    """
    Return the interpolated polygon for frame_idx, or None if the object
    is outside / before first appearance / after last keyframe.
    """
    kfs = track.keyframes
    if not kfs or frame_idx < kfs[0].frame_idx or frame_idx > kfs[-1].frame_idx:
        return None

    prev_kf: _Keyframe | None = None
    next_kf: _Keyframe | None = None
    for kf in kfs:
        if kf.frame_idx <= frame_idx:
            prev_kf = kf
        if kf.frame_idx >= frame_idx and next_kf is None:
            next_kf = kf

    if prev_kf is None:
        return None

    # Exact match on this keyframe
    if prev_kf.frame_idx == frame_idx:
        return None if prev_kf.outside else prev_kf.points

    # Interpolate between prev and next
    if next_kf is None or next_kf.outside:
        return None

    span = next_kf.frame_idx - prev_kf.frame_idx
    t    = (frame_idx - prev_kf.frame_idx) / span if span > 0 else 0.0
    return _lerp_polygon(prev_kf.points, next_kf.points, t)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frame(
    frame_bgr: np.ndarray,
    tracks: list[_Track],
    frame_idx: int,
    opacity: float,
    draw_labels: bool,
) -> np.ndarray:
    """Overlay all track polygons onto frame_bgr (returns a new image)."""
    img     = frame_bgr.copy()
    overlay = img.copy()

    sorted_tracks = sorted(tracks, key=lambda t: t.z_order)

    for track in sorted_tracks:
        pts = get_polygon_at_frame(track, frame_idx)
        if pts is None:
            continue
        color = LABEL_COLORS.get(track.label, _DEFAULT_COLOR)
        ipts  = pts.astype(np.int32)
        cv2.fillPoly(overlay, [ipts], color)
        cv2.polylines(overlay, [ipts], isClosed=True, color=(0, 0, 0), thickness=1)

    cv2.addWeighted(overlay, opacity, img, 1.0 - opacity, 0, img)

    if draw_labels:
        for track in sorted_tracks:
            pts = get_polygon_at_frame(track, frame_idx)
            if pts is None:
                continue
            color = LABEL_COLORS.get(track.label, _DEFAULT_COLOR)
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            (tw, th), _ = cv2.getTextSize(
                track.label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(img, (cx - 2, cy - th - 2), (cx + tw + 2, cy + 2),
                          (0, 0, 0), -1)
            cv2.putText(img, track.label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def load_frames(frames_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted (frame_index, path) pairs from a directory of JPEG/PNG files."""
    exts = {".jpg", ".jpeg", ".png"}
    entries: list[tuple[int, Path]] = []
    for p in frames_dir.iterdir():
        if p.suffix.lower() not in exts:
            continue
        try:
            entries.append((int(p.stem), p))
        except ValueError:
            logger.warning("Skipping non-integer frame file: %s", p.name)
    entries.sort(key=lambda e: e[0])
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize(args: argparse.Namespace) -> None:
    clip_name     = args.clip_name
    xml_path      = Path(args.xml_path)
    frames_dir    = Path(args.clips_dir) / "frames" / clip_name
    opacity       = args.opacity
    fps           = args.fps
    draw_labels   = not args.no_labels

    # Resolve output destinations
    output_frames_dir: Path | None = None
    output_video_path: Path | None = None

    if args.output_frames:
        output_frames_dir = Path(args.output_frames)
    elif not args.output_video:
        # Default: write frames to standard debug location
        output_frames_dir = (
            Path(args.annotation_dir) / "debug_frames" / clip_name
        )

    if args.output_video:
        output_video_path = Path(args.output_video)

    # Validate inputs
    if not xml_path.exists():
        logger.error("XML not found: %s", xml_path)
        sys.exit(1)
    if not frames_dir.exists():
        logger.error("Frames directory not found: %s", frames_dir)
        sys.exit(1)

    tracks = parse_tracks(xml_path)
    if not tracks:
        logger.warning("No tracks found in %s — nothing to render.", xml_path)
        return

    frame_entries = load_frames(frames_dir)
    if not frame_entries:
        logger.error("No image files found in %s", frames_dir)
        sys.exit(1)
    logger.info("Found %d frames in %s", len(frame_entries), frames_dir)

    # Setup outputs
    if output_frames_dir:
        output_frames_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Frame output dir : %s", output_frames_dir)

    writer: cv2.VideoWriter | None = None
    if output_video_path:
        for _, fpath in frame_entries:
            probe = cv2.imread(str(fpath))
            if probe is not None:
                h, w = probe.shape[:2]
                break
        else:
            logger.error("Cannot read any frame to determine resolution.")
            sys.exit(1)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))
        logger.info(
            "Video output     : %s  (%dx%d @ %.1f fps)",
            output_video_path, w, h, fps,
        )

    # Render loop
    rendered = 0
    for frame_idx, frame_path in frame_entries:
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            logger.warning("Cannot read frame %d (%s) — skipping.", frame_idx, frame_path)
            continue

        vis = render_frame(bgr, tracks, frame_idx, opacity, draw_labels)

        if output_frames_dir:
            cv2.imwrite(
                str(output_frames_dir / f"{frame_idx:05d}.jpg"),
                vis,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
        if writer is not None:
            writer.write(vis)

        rendered += 1

    if writer is not None:
        writer.release()
        logger.info("Video saved : %s", output_video_path)
    if output_frames_dir:
        logger.info("Frames saved: %s  (%d files)", output_frames_dir, rendered)

    logger.info("Done. Rendered %d / %d frames.", rendered, len(frame_entries))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render CVAT 1.1 track-format annotations onto clip frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--clip-name", required=True,
        help="Clip name (e.g. right_MOVI0017_0001). Used to resolve default paths.",
    )
    p.add_argument(
        "--annotation-dir",
        default="data/interim/auto_annotations",
        help="Root annotation output dir (same as --output-dir in run_auto_annotation.py).",
    )
    p.add_argument(
        "--xml-path", default=None,
        help=(
            "Path to annotations.xml. "
            "Defaults to <annotation-dir>/cvat_export/<clip-name>/annotations.xml."
        ),
    )
    p.add_argument(
        "--clips-dir",
        default="data/interim/choosed_clips_v5-1",
        help="Root clips dir; frames read from <clips-dir>/frames/<clip-name>/.",
    )
    p.add_argument(
        "--output-frames", default=None, metavar="DIR",
        help=(
            "Write annotated JPEG frames to DIR. "
            "Default (when --output-video is also absent): "
            "<annotation-dir>/debug_frames/<clip-name>/."
        ),
    )
    p.add_argument(
        "--output-video", default=None, metavar="PATH",
        help="Write annotated MP4 to PATH (e.g. out/right_MOVI0017_0001.mp4).",
    )
    p.add_argument(
        "--fps", type=float, default=5.0,
        help="Frames per second for the output video.",
    )
    p.add_argument(
        "--opacity", type=float, default=0.35,
        help="Polygon fill opacity [0.0 – 1.0].",
    )
    p.add_argument(
        "--no-labels", action="store_true",
        help="Suppress per-polygon label text overlay.",
    )

    args = p.parse_args()

    # Resolve default xml-path
    if args.xml_path is None:
        args.xml_path = str(
            Path(args.annotation_dir)
            / "cvat_export"
            / args.clip_name
            / "annotations.xml"
        )

    return args


if __name__ == "__main__":
    visualize(_parse_args())