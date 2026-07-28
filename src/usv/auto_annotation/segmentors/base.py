"""Abstract base class for instance (thing) segmentors."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class InstanceSegmentorBase(ABC):
    """
    Interface contract for box-prompted instance segmentors.

    Implementors: SAM2Segmentor (cpu-sam2 mode).

    Used in Stage 3 of the cpu-sam2 pipeline: receives bounding boxes
    from Florence2Detector on the keyframe and returns one binary mask
    per box, at original frame resolution.

    Return contract for segment()
    ------------------------------
    Returns a list of H×W uint8 binary masks (values 0 or 1),
    one per input bounding box, in the same order as the input.
    If SAM 2 produces an empty mask for a box, that slot contains
    a zero-filled array — the caller (pipeline.py) discards it by
    checking mask_area() < min_instance_area.
    """

    @abstractmethod
    def segment(
        self,
        frame: np.ndarray,
        bboxes_xyxy: list[list[int]],
    ) -> list[np.ndarray]:
        """
        Box-prompted segmentation on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 array of shape (H, W, 3) at original clip resolution.
        bboxes_xyxy : list[list[int]]
            List of [x1, y1, x2, y2] boxes in original frame pixel coordinates.
            Must not be empty.

        Returns
        -------
        list[np.ndarray]
            One H×W uint8 binary mask per input box, same order.
            Empty list only if bboxes_xyxy was empty.
        """
        ...
