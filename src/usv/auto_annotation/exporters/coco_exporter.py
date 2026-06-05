"""
Export TrackAnnotation list to COCO JSON format for instance segmentation.
Output: one COCO JSON per clip with polygons per instance per frame.
"""
from __future__ import annotations
import json
from pathlib import Path
from usv.auto_annotation.types import ClipData, TrackAnnotation

THING_LABELS = ["Vessel", "Buoy", "LandingMark", "BridgeLight", "Other"]

COCO_CATEGORIES = [
    {"id": i + 1, "name": lbl, "supercategory": "obstacle"}
    for i, lbl in enumerate(THING_LABELS)
]
_LABEL_TO_CAT_ID = {lbl: i + 1 for i, lbl in enumerate(THING_LABELS)}


def export_coco(
    clip_data: ClipData,
    tracks: list[TrackAnnotation],
    output_dir: Path,
) -> Path:
    images, annotations = [], []
    ann_id = 1

    # Строим images-список (один entry на кадр)
    for frame_idx in range(clip_data.n_frames):
        images.append({
            "id": frame_idx,
            "file_name": f"{frame_idx:05d}.jpg",
            "height": clip_data.frame_height,
            "width": clip_data.frame_width,
        })

    # Разворачиваем sparse keyframes в per-frame annotations
    # CVAT-style: между двумя keyframes считаем объект видимым
    for track in tracks:
        cat_id = _LABEL_TO_CAT_ID.get(track.label)
        if cat_id is None:
            continue   # stuff-класс — пропускаем

        # Собираем только non-outside keyframes
        visible_kfs = [kf for kf in track.keyframes if not kf.outside]
        for kf in visible_kfs:
            pts_flat = [coord for pt in kf.points for coord in pt]
            xs = [p[0] for p in kf.points]
            ys = [p[1] for p in kf.points]
            bbox_xywh = [
                min(xs), min(ys),
                max(xs) - min(xs), max(ys) - min(ys),
            ]
            area = bbox_xywh[2] * bbox_xywh[3]
            annotations.append({
                "id": ann_id,
                "image_id": kf.frame_idx,
                "category_id": cat_id,
                "segmentation": [pts_flat],
                "bbox": bbox_xywh,
                "area": area,
                "iscrowd": 0,
                "track_id": track.track_id,   # extra field для трекинга
            })
            ann_id += 1

    coco = {
        "info": {"description": f"USV auto-annotation clip={clip_data.clip_name}"},
        "categories": COCO_CATEGORIES,
        "images": images,
        "annotations": annotations,
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_data.clip_name}_coco.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    return out_path