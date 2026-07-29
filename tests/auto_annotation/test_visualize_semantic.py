from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np

_VISUALIZER_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "visualize_annotations.py"
)

_spec = importlib.util.spec_from_file_location(
    "visualize_annotations",
    _VISUALIZER_PATH,
)
assert _spec is not None
assert _spec.loader is not None

visualization = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(visualization)


def _make_semantic_inputs(tmp_path: Path) -> tuple[Path, Path, str]:
    clip_name = "test_clip"
    clips_dir = tmp_path / "clips"
    annotation_dir = tmp_path / "annotations"

    frames_dir = clips_dir / "frames" / clip_name
    label_maps_dir = annotation_dir / "label_maps" / clip_name
    frames_dir.mkdir(parents=True)
    label_maps_dir.mkdir(parents=True)

    frame = np.full((12, 16, 3), 80, dtype=np.uint8)
    label_map = np.zeros((12, 16), dtype=np.uint8)

    assert cv2.imwrite(str(frames_dir / "00000.jpg"), frame)
    assert cv2.imwrite(str(label_maps_dir / "00000.png"), label_map)

    return clips_dir, annotation_dir, clip_name


def _semantic_args(
    clips_dir: Path,
    annotation_dir: Path,
    clip_name: str,
    *,
    output_frames: Path | None = None,
    output_video: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        clip_name=clip_name,
        annot_mode="semantic",
        annotation_dir=str(annotation_dir),
        clips_dir=str(clips_dir),
        output_frames=str(output_frames) if output_frames else None,
        output_video=str(output_video) if output_video else None,
        xml_path=None,
        coco_path=None,
        fps=5.0,
        opacity=1.0,
        no_labels=True,
        no_track_id=False,
    )


def test_render_semantic_frame_uses_palette() -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    label_map = np.array(
        [
            [0, 1, 2],
            [3, 4, 0],
        ],
        dtype=np.uint8,
    )

    result = visualization.render_semantic_frame(
        frame_bgr=frame,
        label_map=label_map,
        opacity=1.0,
        draw_labels=False,
    )

    for label_id, (color, _) in visualization.SEMANTIC_PALETTE.items():
        assert np.all(result[label_map == label_id] == color)


def test_semantic_visualization_writes_custom_frames(tmp_path: Path) -> None:
    clips_dir, annotation_dir, clip_name = _make_semantic_inputs(tmp_path)
    output_frames = tmp_path / "custom_frames"

    args = _semantic_args(
        clips_dir,
        annotation_dir,
        clip_name,
        output_frames=output_frames,
    )

    visualization.visualize(args)

    rendered = output_frames / "00000.jpg"
    assert rendered.exists()
    assert rendered.stat().st_size > 0


def test_semantic_visualization_writes_video(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clips_dir, annotation_dir, clip_name = _make_semantic_inputs(tmp_path)
    output_video = tmp_path / "semantic_preview.mp4"
    written_frames: list[np.ndarray] = []

    class FakeVideoWriter:
        def __init__(self, path, fourcc, fps, size) -> None:
            self.path = Path(path)
            self.size = size
            self.released = False

        def isOpened(self) -> bool:
            return True

        def write(self, frame: np.ndarray) -> None:
            written_frames.append(frame.copy())

        def release(self) -> None:
            self.path.write_bytes(b"fake-mp4")
            self.released = True

    monkeypatch.setattr(visualization.cv2, "VideoWriter", FakeVideoWriter)
    monkeypatch.setattr(
        visualization.cv2,
        "VideoWriter_fourcc",
        lambda *_: 0,
    )

    args = _semantic_args(
        clips_dir,
        annotation_dir,
        clip_name,
        output_video=output_video,
    )

    visualization.visualize(args)

    assert output_video.exists()
    assert output_video.stat().st_size > 0
    assert len(written_frames) == 1
    assert written_frames[0].shape == (12, 16, 3)