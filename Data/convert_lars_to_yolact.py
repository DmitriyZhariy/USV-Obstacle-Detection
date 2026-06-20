import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def rgb2id(color):
    color = np.asarray(color, dtype=np.uint32)
    if color.ndim == 3:
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return color


def ensure_serializable_rle(rle):
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def build_categories(categories, only_things=True, single_class=False):
    cat_id_map = {}
    coco_categories = []
    next_id = 1

    if single_class:
        for cat in categories:
            if only_things and int(cat.get("isthing", 0)) == 0:
                continue
            cat_id_map[int(cat["id"])] = 1

        coco_categories = [{
            "id": 1,
            "name": "obstacle",
            "supercategory": "obstacle"
        }]
        return coco_categories, cat_id_map

    for cat in categories:
        if only_things and int(cat.get("isthing", 0)) == 0:
            continue

        old_id = int(cat["id"])
        cat_id_map[old_id] = next_id
        coco_categories.append({
            "id": next_id,
            "name": cat["name"],
            "supercategory": cat.get("supercategory", cat["name"])
        })
        next_id += 1

    return coco_categories, cat_id_map


def convert_split(
    panoptic_json_path: Path,
    panoptic_masks_dir: Path,
    images_dir: Path,
    output_json_path: Path,
    only_things=True,
    single_class=False,
    min_area=10
):
    data = load_json(panoptic_json_path)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    image_id_to_info = {int(img["id"]): img for img in images}
    coco_categories, cat_id_map = build_categories(
        categories,
        only_things=only_things,
        single_class=single_class
    )

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for ann in annotations:
        image_id = int(ann["image_id"])
        img_info = image_id_to_info[image_id]

        img_file = Path(img_info["file_name"]).name
        img_path = images_dir / img_file
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        coco_images.append({
            "id": image_id,
            "width": int(img_info["width"]),
            "height": int(img_info["height"]),
            "file_name": img_file
        })

        pan_mask_file = Path(ann["file_name"]).name
        pan_mask_path = panoptic_masks_dir / pan_mask_file
        if not pan_mask_path.exists():
            raise FileNotFoundError(f"Panoptic mask not found: {pan_mask_path}")

        pan_rgb = np.array(Image.open(pan_mask_path).convert("RGB"), dtype=np.uint8)
        pan_ids = rgb2id(pan_rgb)

        for seg in ann["segments_info"]:
            src_cat_id = int(seg["category_id"])
            if src_cat_id not in cat_id_map:
                continue

            segment_id = int(seg["id"])
            mask = (pan_ids == segment_id).astype(np.uint8)

            area = int(mask.sum())
            if area < min_area:
                continue

            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue

            x_min = int(xs.min())
            y_min = int(ys.min())
            x_max = int(xs.max())
            y_max = int(ys.max())
            bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

            rle = mask_utils.encode(np.asfortranarray(mask))
            rle = ensure_serializable_rle(rle)

            coco_annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id_map[src_cat_id],
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
            ann_id += 1

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories
    }

    save_json(coco, output_json_path)

    print(f"Saved: {output_json_path}")
    print(f"Images: {len(coco_images)}")
    print(f"Annotations: {len(coco_annotations)}")
    print("Categories:")
    for c in coco_categories:
        print(f"  {c['id']}: {c['name']}")

    return coco


def main():
    print("Script started")
    parser = argparse.ArgumentParser(description="Convert LaRS panoptic dataset to COCO instance dataset for YOLACT")
    parser.add_argument("--lars_root", type=str, required=True, help="Path to LaRS root folder")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: <LaRS>/yolact_instance)")
    parser.add_argument("--include_stuff", action="store_true", help="Include stuff classes too (usually not needed for YOLACT)")
    parser.add_argument("--single_class", action="store_true", help="Merge all kept classes into one class: obstacle")
    parser.add_argument("--min_area", type=int, default=10, help="Minimum object area in pixels")
    args = parser.parse_args()

    lars_root = Path(args.lars_root)

    train_pan_json = lars_root / "annotations" / "train" / "panoptic_train.json"
    val_pan_json   = lars_root / "annotations" / "val" / "panoptic_val.json"

    train_pan_masks = lars_root / "annotations" / "train" / "panoptic_masks"
    val_pan_masks   = lars_root / "annotations" / "val" / "panoptic_masks"

    train_images = lars_root / "image" / "train" / "images"
    val_images   = lars_root / "image" / "val" / "images"

    output_dir = Path(args.output_dir) if args.output_dir else (lars_root / "yolact_instance")
    output_ann_dir = output_dir / "annotations"
    output_ann_dir.mkdir(parents=True, exist_ok=True)

    if not train_pan_json.exists():
        raise FileNotFoundError(f"Train panoptic json not found: {train_pan_json}")
    if not val_pan_json.exists():
        raise FileNotFoundError(f"Val panoptic json not found: {val_pan_json}")

    print("Converting TRAIN split...")
    convert_split(
        panoptic_json_path=train_pan_json,
        panoptic_masks_dir=train_pan_masks,
        images_dir=train_images,
        output_json_path=output_ann_dir / "instances_train.json",
        only_things=not args.include_stuff,
        single_class=args.single_class,
        min_area=args.min_area
    )

    print("\nConverting VAL split...")
    convert_split(
        panoptic_json_path=val_pan_json,
        panoptic_masks_dir=val_pan_masks,
        images_dir=val_images,
        output_json_path=output_ann_dir / "instances_val.json",
        only_things=not args.include_stuff,
        single_class=args.single_class,
        min_area=args.min_area
    )

    print("\nDone.")
    print(f"Train JSON: {output_ann_dir / 'instances_train.json'}")
    print(f"Val JSON:   {output_ann_dir / 'instances_val.json'}")


if __name__ == "__main__":
    main()