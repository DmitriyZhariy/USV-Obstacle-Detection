"""Abstract base class for stuff (semantic) segmentors."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class StuffSegmentorBase(ABC):
    """
    Interface contract for frame-level stuff segmentors.

    Implementors: SegFormerStuff (cpu-sam2 mode).

    Return contract for segment_frame()
    ------------------------------------
    Returns an H×W uint8 ndarray where each pixel value is a
    **project class ID** (from configs/auto_annotation.yaml labels block),
    not a raw ADE20K ID.

    Pixels with no mapped stuff class → 255 (ignore / unmapped sentinel).
    Pixels that belong to thing classes are NOT masked out here —
    overlap_resolver.py handles thing-over-stuff layering via Z-order.
    """

    @abstractmethod
    def segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run semantic segmentation on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 array of shape (H, W, 3) at original clip resolution.

        Returns
        -------
        np.ndarray
            uint8 array of shape (H, W).
            Values are project class IDs or 255 for unmapped pixels.
        """
        ...
