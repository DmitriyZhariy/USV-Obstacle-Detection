"""Abstract base class for all thing-class detectors."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DetectorBase(ABC):
    """
    Interface contract for frame-level thing detectors.

    Both YOLOv8Detector (cpu-fast) and Florence2Detector (cpu-sam2)
    implement this interface so that pipeline.py can swap them via
    simple mode dispatch without any other code changes.

    Return schema for detect()
    --------------------------
    Each dict in the returned list must contain:

        label      : str              — project class name, e.g. "Vessel"
        class_id   : int              — integer ID from config labels block
        z_order    : int              — fixed Z-order from config
        confidence : float            — detector confidence score [0, 1]
        bbox_xyxy  : tuple[float, float, float, float]
                                      — (x1, y1, x2, y2) in original frame resolution
        mask       : np.ndarray | None
                                      — H×W uint8 binary mask at original resolution,
                                        or None when the detector does not produce masks
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a single BGR uint8 frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 array of shape (H, W, 3) at original clip resolution.

        Returns
        -------
        list[dict]
            Zero or more detection dicts matching the schema above.
            Returns an empty list when nothing is detected.
        """
        ...
