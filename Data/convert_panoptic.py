import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import shutil

LARS_ROOT = r"E:\Education\4 course 2 semester\Practice\panoptic_project\Data\Lars"
OUT_ROOT = r"E:\Education\4 course 2 semester\Practice\panoptic_project\Data\lars_converted"

STUFF_CLASSES = {1: 0, 3: 1, 5: 2}
THING_CLASSES = [11, 12, 13, 14, 15, 16, 17, 19]


def rgb2id(color):
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color = color.astype(np.int32)
    return color[:, :, 0] + color[:, :, 1] * 256 + color[:, :, 2] * 256 * 256


def build_semantic(panoptic, segments_info):
    h, w = panoptic.shape
    semantic = np.zeros((h, w), dtype=np.uint8)

    for seg in segments_info:
        cid = seg["category_id"]
        sid = seg["id"]

        if cid in STUFF_CLASSES:
            semantic[panoptic == sid] = STUFF_CLASSES[cid]

    return semantic


def extract_instances(panoptic, segments_info):
    instances = []

    for seg in segments_info:
        cid = seg["category_id"]
        sid = seg["id"]

        if cid in THING_CLASSES:
            mask = (panoptic == sid).astype(np.uint8)

            if mask.sum() < 50:
                continue

            instances.append((cid, mask))

    return instances


def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []

    for cnt in contours:
        if len(cnt) < 3:
            continue

        cnt = cnt.squeeze()

        if len(cnt.shape) != 2:
            continue

        polygons.append(cnt)

    return polygons


def normalize_polygon(poly, w, h):
    return [(x / w, y / h) for x, y in poly]


def make_dirs(split):
    paths = [
        f"semantic/{split}/images",
        f"semantic/{split}/masks",
        f"yolo/{split}/images",
        f"yolo/{split}/labels",
    ]
    for p in paths:
        os.makedirs(os.path.join(OUT_ROOT, p), exist_ok=True)


def process_split(split):

    print(f"\n🚀 Processing {split}...")

    IMG_DIR = os.path.join(LARS_ROOT, f"image/{split}/images")
    PANOPTIC_MASK_DIR = os.path.join(LARS_ROOT, f"annotations/{split}/panoptic_masks")
    PANOPTIC_JSON = os.path.join(LARS_ROOT, f"annotations/{split}/panoptic_annotations.json")

    make_dirs(split)

    with open(PANOPTIC_JSON) as f:
        data = json.load(f)

    annotations = {ann["image_id"]: ann for ann in data["annotations"]}

    for img in tqdm(data["images"]):

        img_id = img["id"]
        file_name = img["file_name"]

        img_path = os.path.join(IMG_DIR, file_name)
        mask_path = os.path.join(PANOPTIC_MASK_DIR, file_name.replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        if image is None:
            continue

        panoptic_rgb = cv2.imread(mask_path)
        if panoptic_rgb is None:
            continue

        panoptic = rgb2id(panoptic_rgb)

        if img_id not in annotations:
            continue

        segments_info = annotations[img_id]["segments_info"]

        h, w = panoptic.shape

        # ===== SEMANTIC =====
        semantic = build_semantic(panoptic, segments_info)

        cv2.imwrite(
            os.path.join(OUT_ROOT, f"semantic/{split}/masks", file_name.replace(".jpg", ".png")),
            semantic
        )

        shutil.copy(img_path, os.path.join(OUT_ROOT, f"semantic/{split}/images", file_name))

        # ===== YOLO =====
        instances = extract_instances(panoptic, segments_info)

        yolo_lines = []

        for cid, mask in instances:

            polygons = mask_to_polygons(mask)

            for poly in polygons:
                poly = normalize_polygon(poly, w, h)

                flat = [str(coord) for point in poly for coord in point]

                cls_id = THING_CLASSES.index(cid)

                line = str(cls_id) + " " + " ".join(flat)
                yolo_lines.append(line)

        label_path = os.path.join(OUT_ROOT, f"yolo/{split}/labels", file_name.replace(".jpg", ".txt"))

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        shutil.copy(img_path, os.path.join(OUT_ROOT, f"yolo/{split}/images", file_name))


def create_yaml():
    yaml_text = f"""
path: {OUT_ROOT}/yolo

train: train/images
val: val/images

names:
  0: boat
  1: rowboat
  2: paddleboard
  3: buoy
  4: swimmer
  5: animal
  6: float
  7: other
"""
    with open(os.path.join(OUT_ROOT, "yolo_dataset.yaml"), "w") as f:
        f.write(yaml_text)


def convert():
    process_split("train")
    process_split("val")
    create_yaml()
    print("\n✅ Conversion finished!")


if __name__ == "__main__":
    convert()