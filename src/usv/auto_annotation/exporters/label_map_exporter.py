"""
Label map exporter for semantic segmentation pipeline.

Converts SegFormerStuff output (H×W uint8 project class_id map)
into PNG label maps compatible with ADE20K / Cityscapes training format.

Output structure:
    <output_dir>/
        <clip_name>/
            00000.png   ← per-pixel class_id, uint8
            00001.png
            ...
            palette/
                00000_vis.jpg   ← coloured overlay for visual QC
                ...

class_id palette (matches configs/auto_annotation.yaml):
    0   Water   (30, 144, 255) BGR
    1   Sky     (135, 206, 235)
    2   Land    (34, 139, 34)
    3   Pier    (139, 69, 19)
    4   Bridge  (112, 128, 144)
    255 Void / unmapped → black in vis overlay

PNG files use single-channel mode (mode='L') — standard for label maps.
Training frameworks (mmsegmentation, HuggingFace Trainer) read class_id
directly from pixel values.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR colours for visualisation — matches visualize_annotations.py palette
_CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: ( 30, 144, 255),   # Water
    1: (135, 206, 235),   # Sky
    2: ( 34, 139,  34),   # Land
    3: (139,  69,  19),   # Pier
    4: (112, 128, 144),   # Bridge
}
_VOID_COLOR = (0, 0, 0)   # unmapped → black


def export_label_maps(
    clip_name: str,
    label_maps: list[np.ndarray],
    output_dir: Path,
    save_vis: bool = True,
) -> Path:
    """
    Write per-frame PNG label maps to disk.

    Parameters
    ----------
    clip_name : str
        Used as subdirectory name under output_dir.
    label_maps : list[np.ndarray]
        List of H×W uint8 arrays from SegFormerStuff.segment_frame().
        Index == frame index (0-based).
    output_dir : Path
        Root export directory. Maps go to output_dir/<clip_name>/.
    save_vis : bool
        If True, also write coloured JPEG overlays to
        output_dir/<clip_name>/palette/.

    Returns
    -------
    Path
        Path to output_dir/<clip_name>/ directory.
    """
    clip_dir = output_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    vis_dir: Path | None = None
    if save_vis:
        vis_dir = clip_dir / "palette"
        vis_dir.mkdir(exist_ok=True)

    n = len(label_maps)
    for frame_idx, label_map in enumerate(label_maps):
        # ── PNG label map (single-channel, values = class_id) ──────────
        png_path = clip_dir / f"{frame_idx:05d}.png"
        cv2.imwrite(str(png_path), label_map)

        # ── Colour visualisation JPEG ───────────────────────────────────
        if vis_dir is not None:
            vis = _colorize(label_map)
            jpg_path = vis_dir / f"{frame_idx:05d}_vis.jpg"
            cv2.imwrite(str(jpg_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

    logger.info(
        "label_map_exporter: %d frames written to %s%s",
        n,
        clip_dir,
        f" + {n} vis frames" if save_vis else "",
    )
    return clip_dir


def _colorize(label_map: np.ndarray) -> np.ndarray:
    """Convert class_id map to BGR colour image for visual QC."""
    h, w = label_map.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in _CLASS_COLORS_BGR.items():
        vis[label_map == class_id] = color
    # 255 (void) stays black
    return vis