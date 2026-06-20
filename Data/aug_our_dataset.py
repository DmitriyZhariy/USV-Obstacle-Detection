import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import albumentations as A



DATASET_ROOT = Path(
    r"E:/Education/4 course 2 semester/Diploma/panoptic_project/data/our_dataset"
)


AUG_PER_IMAGE = 8

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OVERWRITE = False

MIN_CONTOUR_AREA = 20.0
MIN_POINTS = 3




def get_transform(img_h, img_w):
    return A.Compose([
        A.RandomResizedCrop(
            size=(img_h, img_w),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.6,
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.04,
            scale_limit=0.08,
            rotate_limit=8,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.6,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.20,
            contrast_limit=0.20,
            p=0.4,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3,
        ),
    ])




def list_image_label_pairs(split_dir: Path):
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    pairs = []

    for img_path in img_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        label_path = lbl_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            # Не аугментируем уже созданные aug-файлы повторно
            if "_aug_" not in img_path.stem:
                pairs.append((img_path, label_path))

    return pairs


def read_yolo_segmentation_label(label_path: Path):
    objects = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            cls_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))

            if len(coords) < 6 or len(coords) % 2 != 0:
                continue

            objects.append((cls_id, coords))

    return objects


def write_yolo_segmentation_label(label_path: Path, objects):
    lines = []
    for cls_id, coords in objects:
        if len(coords) < 6:
            continue
        line = " ".join([str(cls_id)] + [f"{c:.6f}" for c in coords])
        lines.append(line)

    if lines:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def polygon_to_mask(coords, img_w, img_h):
    pts = []
    for i in range(0, len(coords), 2):
        x = int(round(np.clip(coords[i], 0.0, 1.0) * (img_w - 1)))
        y = int(round(np.clip(coords[i + 1], 0.0, 1.0) * (img_h - 1)))
        pts.append([x, y])

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if len(pts) >= 3:
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)

    return mask


def objects_to_masks(objects, img_w, img_h):
    masks = []
    class_ids = []

    for cls_id, coords in objects:
        mask = polygon_to_mask(coords, img_w, img_h)
        if mask.sum() > 0:
            masks.append(mask)
            class_ids.append(cls_id)

    return masks, class_ids


def mask_to_polygons(mask, epsilon_ratio=0.002):
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        eps = epsilon_ratio * peri
        approx = cv2.approxPolyDP(cnt, eps, True)

        if len(approx) < MIN_POINTS:
            continue

        polygons.append(approx.reshape(-1, 2))

    return polygons


def masks_to_yolo_objects(masks, class_ids, img_w, img_h):
    objects = []

    for mask, cls_id in zip(masks, class_ids):
        polygons = mask_to_polygons(mask)

        for poly in polygons:
            if len(poly) < MIN_POINTS:
                continue

            coords = []
            for x, y in poly:
                xn = np.clip(x / img_w, 0.0, 1.0)
                yn = np.clip(y / img_h, 0.0, 1.0)
                coords.extend([float(xn), float(yn)])

            if len(coords) >= 6:
                objects.append((cls_id, coords))

    return objects



def augment_split(split: str):
    split_dir = DATASET_ROOT / split
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[WARN] split {split} not found, skipping")
        return

    print(f"Processing split: {split}")

    out_img_dir = img_dir
    out_lbl_dir = lbl_dir

    if OVERWRITE:
        print(f"[INFO] OVERWRITE=True: clearing {split}...")
        shutil.rmtree(img_dir)
        shutil.rmtree(lbl_dir)
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_image_label_pairs(split_dir)
    print(f"Found {len(pairs)} original image/label pairs in {split}")

    counter = 0

    for img_path, label_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        h, w = img.shape[:2]
        objects = read_yolo_segmentation_label(label_path)

        if not objects:
            continue

        if not OVERWRITE:
            dst_img = out_img_dir / img_path.name
            dst_lbl = out_lbl_dir / label_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            if not dst_lbl.exists():
                shutil.copy2(label_path, dst_lbl)

        masks, class_ids = objects_to_masks(objects, w, h)
        if not masks:
            continue

        for i in range(AUG_PER_IMAGE):
            transform = get_transform(h, w)
            transformed = transform(image=img, masks=masks)

            aug_img = transformed["image"]
            aug_masks = transformed["masks"]

            ah, aw = aug_img.shape[:2]
            aug_objects = masks_to_yolo_objects(aug_masks, class_ids, aw, ah)

            if not aug_objects:
                continue

            base = img_path.stem
            ext = img_path.suffix
            aug_name = f"{base}_aug_{i+1:02d}{ext}"
            aug_label_name = f"{base}_aug_{i+1:02d}.txt"

            out_img_path = out_img_dir / aug_name
            out_lbl_path = out_lbl_dir / aug_label_name

            cv2.imwrite(str(out_img_path), aug_img)
            write_yolo_segmentation_label(out_lbl_path, aug_objects)

            counter += 1

        if counter and counter % 100 == 0:
            print(f"{split}: generated {counter} augmented images so far...")

    print(f"{split}: total augmented images created: {counter}")


def main():
    random.seed(42)
    np.random.seed(42)

    augment_split("train")


if __name__ == "__main__":
    main()