"""
Pure NumPy/OpenCV utilities for mask-polygon conversion and IoU computation.
No model dependencies. Used by both cpu-fast and cpu-sam2 paths.
"""
from __future__ import annotations
import cv2
import numpy as np


def mask_area(mask: np.ndarray) -> int:
    """Return the number of non-zero pixels in a binary mask (H×W uint8)."""
    return int(np.count_nonzero(mask))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union between two binary masks of the same shape."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union > 0 else 0.0


def bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """IoU between two xyxy bounding boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def mask_to_polygon(
    mask: np.ndarray,
    epsilon_ratio: float = 0.005,
    min_area: int = 64,
) -> list[tuple[float, float]] | None:
    """
    Convert a binary mask to a simplified polygon via cv2.approxPolyDP.

    min_area=64 implements the annotation manual rule: objects < 8×8 px
    are Void (manual only) and must not be auto-annotated.

    Args:
        mask:          H×W uint8. Accepts 0/1 or 0/255 — both normalised.
        epsilon_ratio: approxPolyDP epsilon = ratio × contour perimeter.
        min_area:      Skip contours below this area in px².

    Returns:
        List of (x, y) float tuples, or None if no valid contour found.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    elif mask.max() == 1:
        mask = mask * 255

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None

    epsilon = epsilon_ratio * cv2.arcLength(contour, closed=True)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)

    if len(approx) < 3:
        return None

    return [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
