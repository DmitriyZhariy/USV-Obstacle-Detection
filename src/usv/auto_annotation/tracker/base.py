"""Abstract base class for all instance trackers."""
from __future__ import annotations

from abc import ABC, abstractmethod

from usv.auto_annotation.types import TrackAnnotation


class TrackerBase(ABC):
    """
    Interface contract for clip-level instance trackers.

    Both IoUTracker (cpu-fast) and SAM2Tracker (cpu-sam2) implement
    this interface. The calling pattern is always:

        tracker = ConcreteTracker(...)
        for frame_idx, frame in enumerate(clip.frames):
            dets = detector.detect(frame)
            tracker.update(frame_idx, frame, dets)
        tracks = tracker.finalize()

    Note: frame is passed to update() because SAM2Tracker needs raw
    pixels for propagation, while IoUTracker ignores it. Both must
    accept the argument.
    """

    @abstractmethod
    def update(
        self,
        frame_idx: int,
        frame: "np.ndarray",
        detections: list[dict],
    ) -> None:
        """
        Feed one frame into the tracker.

        Parameters
        ----------
        frame_idx : int
            0-based index of this frame in the clip.
        frame : np.ndarray
            BGR uint8 array (H, W, 3) at original resolution.
            Used by SAM2Tracker for propagation; ignored by IoUTracker.
        detections : list[dict]
            Output of DetectorBase.detect() for this frame.
        """
        ...

    @abstractmethod
    def finalize(self) -> list[TrackAnnotation]:
        """
        Close all open tracks and return the complete annotation list.

        Must be called exactly once, after all frames have been
        passed to update(). Calling update() after finalize() is
        undefined behaviour.

        Returns
        -------
        list[TrackAnnotation]
            One TrackAnnotation per tracked object instance,
            ready for cvat_xml_exporter.py.
        """
        ...
