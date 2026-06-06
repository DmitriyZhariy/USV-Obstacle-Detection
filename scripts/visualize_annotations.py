"""
Render CVAT 1.1 track-format annotations onto a video clip.

Reads the annotations.xml produced by run_auto_annotation.py (cpu-sam2 mode),
interpolates polygons between sparse keyframes, and writes either:
  - a debug JPEG strip  (default, --output-frames)
  - an MP4 video        (--output-video)
  - both                (both flags together)

Supports both annotation formats:
  --annot-mode panoptic  (default) → reads CVAT XML (annotations.xml)
  --annot-mode instance            → reads COCO JSON (*_coco.json)

Usage – panoptic (default):
    uv run python -m scripts.visualize_annotations \
        --clip-name right_MOVI0017_0001

Usage – instance (COCO JSON):
    uv run python -m scripts.visualize_annotations \
        --clip-name right_MOVI0017_0001 \
        --annot-mode instance

Usage – explicit paths + video output:
    uv run python -m scripts.visualize_annotations \
        --clip-name right_MOVI0017_0001 \
        --output-video out/right_MOVI0017_0001.mp4 \
        --fps 5 \
        --opacity 0.35
"""
from __future__ import annotations

import argparse
import json
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
# Label palette (BGR)
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

# Semantic label map: id → (BGR color, name)
SEMANTIC_PALETTE: dict[int, tuple[tuple[int, int, int], str]] = {
    0: ((255, 144,  30), "Water"),
    1: ((235, 206, 135), "Sky"),
    2: (( 34, 139,  34), "Land"),
    3: (( 19,  69, 139), "Pier"),
    4: ((144, 128, 112), "Bridge"),
}

def render_semantic_frame(
    frame_bgr: np.ndarray,
    label_map: np.ndarray,   # H×W uint8, значения 0..4
    opacity: float,
    draw_labels: bool,
) -> np.ndarray:
    img     = frame_bgr.copy()
    overlay = img.copy()
    h, w    = img.shape[:2]

    for label_id, (color, name) in SEMANTIC_PALETTE.items():
        mask = label_map == label_id
        if not mask.any():
            continue
        overlay[mask] = color

    cv2.addWeighted(overlay, opacity, img, 1.0 - opacity, 0, img)

    if draw_labels:
        for label_id, (color, name) in SEMANTIC_PALETTE.items():
            mask = label_map == label_id
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (cx - 2, cy - th - 2), (cx + tw + 2, cy + 2), (0, 0, 0), -1)
            cv2.putText(img, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return img

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class _Keyframe(NamedTuple):
    frame_idx: int
    points: np.ndarray      # shape (N, 2), float32
    outside: bool


class _Track(NamedTuple):
    label: str
    z_order: int
    keyframes: list[_Keyframe]
    track_id: int = 0


# ---------------------------------------------------------------------------
# CVAT XML parser (без изменений от оригинала)
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


def parse_tracks_xml(xml_path: Path) -> list[_Track]:
    """Parse all <track> elements from a CVAT 1.1 XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tracks: list[_Track] = []
    for track_el in root.findall(".//track"):
        label    = track_el.get("label", "")
        z_order  = int(track_el.get("z_order", "0"))
        track_id = int(track_el.get("id", "0"))
        kfs: list[_Keyframe] = []

        for poly_el in track_el.findall("polygon"):
            frame_idx = int(poly_el.get("frame", "0"))
            outside   = poly_el.get("outside", "0") == "1"
            points    = _parse_points(poly_el.get("points", ""))
            kfs.append(_Keyframe(frame_idx=frame_idx, points=points, outside=outside))

        kfs.sort(key=lambda k: k.frame_idx)
        if kfs:
            tracks.append(_Track(
                label=label, z_order=z_order,
                track_id=track_id, keyframes=kfs,
            ))

    logger.info("Parsed %d tracks from %s", len(tracks), xml_path.name)
    return tracks


# ---------------------------------------------------------------------------
# COCO JSON parser
# ---------------------------------------------------------------------------

def parse_tracks_coco(coco_path: Path) -> list[_Track]:
    """
    Parse COCO JSON produced by coco_exporter.py into _Track list.
    Each annotation is an exact keyframe (no interpolation needed).
    track_id extra field groups annotations into tracks.
    """
    from collections import defaultdict

    data = json.loads(coco_path.read_text(encoding="utf-8"))
    cat_id_to_name: dict[int, str] = {
        c["id"]: c["name"] for c in data.get("categories", [])
    }

    groups: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        tid = ann.get("track_id", ann["id"])
        groups[tid].append(ann)

    tracks: list[_Track] = []
    for track_id, anns in groups.items():
        label = cat_id_to_name.get(anns[0]["category_id"], "Unknown")
        kfs: list[_Keyframe] = []
        for ann in anns:
            seg = ann.get("segmentation", [[]])
            if not seg or not seg[0]:
                continue
            flat = seg[0]
            pts = np.array(
                [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)],
                dtype=np.float32,
            )
            kfs.append(_Keyframe(frame_idx=ann["image_id"], points=pts, outside=False))
        kfs.sort(key=lambda k: k.frame_idx)
        if kfs:
            tracks.append(_Track(label=label, z_order=30, keyframes=kfs, track_id=track_id))

    logger.info(
        "Parsed %d tracks (%d annotations) from %s",
        len(tracks), len(data.get("annotations", [])), coco_path.name,
    )
    return tracks


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def _lerp_polygon(pts_a: np.ndarray, pts_b: np.ndarray, t: float) -> np.ndarray:
    if pts_a.shape == pts_b.shape:
        return (1.0 - t) * pts_a + t * pts_b
    return pts_a if t < 0.5 else pts_b


def get_polygon_at_frame(track: _Track, frame_idx: int) -> np.ndarray | None:
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
    if prev_kf.frame_idx == frame_idx:
        return None if prev_kf.outside else prev_kf.points
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
    show_track_id: bool = True,
) -> np.ndarray:
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
            # Показываем "Label #id"
            text = f"{track.label} #{track.track_id}" if show_track_id else track.label
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (cx - 2, cy - th - 2), (cx + tw + 2, cy + 2),
                          (0, 0, 0), -1)
            cv2.putText(img, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def load_frames(frames_dir: Path) -> list[tuple[int, Path]]:
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
    clip_name   = args.clip_name
    annot_mode  = args.annot_mode
    frames_dir  = Path(args.clips_dir) / "frames" / clip_name
    opacity     = args.opacity
    fps         = args.fps
    draw_labels = not args.no_labels

    # Резолв путей к аннотациям (делается здесь, а не в _parse_args,
    # чтобы работать при --all когда clip_name меняется в цикле)
    xml_path = args.xml_path or str(
        Path(args.annotation_dir) / "cvat_export" / clip_name / "annotations.xml"
    )
    coco_path = args.coco_path or str(
        Path(args.annotation_dir) / "cvat_export" / f"{clip_name}_coco.json"
    )

    if annot_mode == "semantic":
        label_maps_dir = Path(args.annotation_dir) / "label_maps" / clip_name
        if not label_maps_dir.exists():
            logger.error("Label maps not found: %s", label_maps_dir)
            sys.exit(1)

        out_dir = Path(args.annotation_dir) / "debug_frames" / f"{clip_name}_semantic"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Semantic output: %s", out_dir)

        frame_entries = load_frames(frames_dir)
        label_paths   = sorted(label_maps_dir.glob("*.png"), key=lambda p: int(p.stem))

        if len(frame_entries) != len(label_paths):
            logger.warning(
                "Frame count (%d) != label map count (%d) — will render min of both",
                len(frame_entries), len(label_paths),
            )

        rendered = 0
        for (frame_idx, frame_path), label_path in zip(frame_entries, label_paths):
            bgr = cv2.imread(str(frame_path))
            label_map = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if bgr is None or label_map is None:
                logger.warning("Skipping frame %d — cannot read files.", frame_idx)
                continue

            vis = render_semantic_frame(bgr, label_map, opacity, draw_labels)
            cv2.imwrite(
                str(out_dir / f"{frame_idx:05d}.jpg"), vis,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            rendered += 1

        logger.info("Done. Rendered %d frames → %s", rendered, out_dir)
        return  # ← ранний выход, остальная логика panoptic/instance не нужна

    # ── Загружаем треки в зависимости от annot_mode ──────────────────────
    if annot_mode == "instance":
        coco_path = Path(coco_path)
        if not coco_path.exists():
            logger.error("COCO JSON not found: %s", coco_path)
            sys.exit(1)
        tracks = parse_tracks_coco(coco_path)
    else:
        # panoptic / default → CVAT XML
        xml_path = Path(xml_path)
        if not xml_path.exists():
            logger.error("XML not found: %s", xml_path)
            sys.exit(1)
        tracks = parse_tracks_xml(xml_path)

    if not tracks:
        logger.warning("No tracks found — nothing to render.")
        return

    if not frames_dir.exists():
        logger.error("Frames directory not found: %s", frames_dir)
        sys.exit(1)

    # ── Resolve output destinations ───────────────────────────────────────
    output_frames_dir: Path | None = None
    output_video_path: Path | None = None

    if args.output_frames:
        output_frames_dir = Path(args.output_frames)
    elif not args.output_video:
        output_frames_dir = (
            Path(args.annotation_dir) / "debug_frames" / f"{clip_name}_{annot_mode}"
        )

    if args.output_video:
        output_video_path = Path(args.output_video)

    frame_entries = load_frames(frames_dir)
    if not frame_entries:
        logger.error("No image files found in %s", frames_dir)
        sys.exit(1)
    logger.info("Found %d frames in %s", len(frame_entries), frames_dir)

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

    # ── Render loop ───────────────────────────────────────────────────────
    rendered = 0
    for frame_idx, frame_path in frame_entries:
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            logger.warning("Cannot read frame %d — skipping.", frame_idx)
            continue

        vis = render_frame(
            bgr, tracks, frame_idx, opacity, draw_labels,
            show_track_id=not args.no_track_id,
        )

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
        description="Render auto-annotation results onto clip frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--clip-name", default=None,
        help="Single clip to visualize. Omit to process all clips (see --all).")
    p.add_argument(
        "--all", action="store_true",
        help=(
            "Process all clips found in <annotation-dir>/cvat_export/ "
            "(panoptic: subdirectories with annotations.xml; "
            "instance: *_coco.json files)."
        ),
    )
    p.add_argument(
        "--annot-mode",
        choices=["panoptic", "instance", "semantic"],
        default="panoptic",
        help=(
            "panoptic = read CVAT XML (annotations.xml). "
            "semantic = read CVAT XML (annotations.xml). "
            "instance = read COCO JSON (*_coco.json)."
        ),
    )
    p.add_argument(
        "--annotation-dir",
        default="data/interim/auto_annotations",
    )
    p.add_argument(
        "--xml-path", default=None,
        help="Path to annotations.xml. "
             "Default: <annotation-dir>/cvat_export/<clip-name>/annotations.xml.",
    )
    p.add_argument(
        "--coco-path", default=None,
        help="Path to COCO JSON. "
             "Default: <annotation-dir>/cvat_export/<clip-name>_coco.json.",
    )
    p.add_argument(
        "--clips-dir",
        default="data/interim/choosed_clips_v5-1",
    )
    p.add_argument("--output-frames", default=None, metavar="DIR")
    p.add_argument("--output-video", default=None, metavar="PATH")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--opacity", type=float, default=0.35)
    p.add_argument("--no-labels", action="store_true")
    p.add_argument(
        "--no-track-id", action="store_true",
        help="Hide track ID suffix in label text (show only label name).",
    )

    args = p.parse_args()

    # if args.coco_path is None:
    #     args.coco_path = str(
    #         Path(args.annotation_dir)
    #         / "cvat_export"
    #         / f"{args.clip_name}_coco.json"
    #     )
    # # Resolve default paths
    # if args.xml_path is None:
    #     args.xml_path = str(
    #         Path(args.annotation_dir)
    #         / "cvat_export"
    #         / args.clip_name
    #         / "annotations.xml"
    #     )
    # if args.coco_path is None:
    #     args.coco_path = str(
    #         Path(args.annotation_dir)
    #         / "cvat_export"
    #         / f"{args.clip_name}_coco.json"
    #     )

    return args


if __name__ == "__main__":
    args = _parse_args()

    if not args.all and not args.clip_name:
        import sys
        print("error: specify --clip-name NAME or --all", file=sys.stderr)
        sys.exit(1)

    if args.all:
        annot_dir = Path(args.annotation_dir)
        cvat_dir = annot_dir / "cvat_export"
        if args.annot_mode == "panoptic":
            clip_names = sorted(
                d.name for d in cvat_dir.iterdir()
                if d.is_dir() and (d / "annotations.xml").exists()
            )
        elif args.annot_mode == "semantic":
            clip_names = sorted(
                d.name for d in (annot_dir / "label_maps").iterdir()
                if d.is_dir()
            )
        else:
            clip_names = sorted(
                p.stem.removesuffix("_coco")
                for p in cvat_dir.glob("*_coco.json")
            )

        if not clip_names:
            logger.warning("No clips found in %s for mode=%s", cvat_dir, args.annot_mode)
            sys.exit(0)

        logger.info("--all: %d clips to visualize", len(clip_names))
        for clip_name in clip_names:
            logger.info("── %s ──", clip_name)
            args.clip_name = clip_name
            # Обновить пути под текущий клип
            args.xml_path = str(
                Path(args.annotation_dir) / "cvat_export" / clip_name / "annotations.xml"
            )
            args.coco_path = str(
                Path(args.annotation_dir) / "cvat_export" / f"{clip_name}_coco.json"
            )
            # output_frames сбрасываем — пусть каждый клип пишется в свою папку по умолчанию
            args.output_frames = None
            args.output_video = None
            visualize(args)
    else:
        visualize(args)