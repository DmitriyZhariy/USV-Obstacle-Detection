import shutil
from pathlib import Path
import cv2
import numpy as np


# =========================================
# НАСТРОЙКИ
# =========================================
SOURCE_DATASET = Path(r"E:\Education\4 course 2 semester\Diploma\panoptic_project\data\LaRS_fusion\instance_yolo")
OUTPUT_DATASET = Path(r"E:\Education\4 course 2 semester\Diploma\panoptic_project\data\new_LaRS")

PADDING = 25
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# Итоговые классы
CLASS_NAMES = {
    0: "boat",
    1: "buoy",
    2: "other",
}

# Remap старых классов в новые
# None = удалить объект полностью
CLASS_REMAP = {
    0: 0,      # boat -> boat
    1: 0,      # rowboat -> boat
    2: 2,      # paddleboard -> other
    3: 1,      # buoy -> buoy
    4: 2,      # swimmer -> other
    5: 2,      # animal -> other
    6: None,   # float -> удалить
    7: 2,      # other -> other
}

OLD_CLASS_NAMES = {
    0: "boat",
    1: "rowboat",
    2: "paddleboard",
    3: "buoy",
    4: "swimmer",
    5: "animal",
    6: "float",
    7: "other",
}

# Аугментация только для новых классов buoy и other
AUGS_PER_INSTANCE = {
    0: 0,   # boat
    1: 4,   # buoy
    2: 4,   # other
}

AUGMENTATION_POOL = [
    "base",
    "pad_plus_10",
    "pad_minus_10",
    "hflip",
    "bright_up",
    "bright_down",
    "contrast_up",
    "blur3",
]


# =========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================
def ensure_structure(root: Path):
    for split in ["train", "val"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)


def find_image(images_dir: Path, stem: str):
    for ext in IMAGE_EXTENSIONS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_yolo_seg_line(line: str):
    parts = line.strip().split()
    if len(parts) < 7:
        return None

    cls_id = int(float(parts[0]))
    coords = list(map(float, parts[1:]))

    if len(coords) % 2 != 0:
        return None

    xs = np.array(coords[0::2], dtype=np.float32)
    ys = np.array(coords[1::2], dtype=np.float32)
    return cls_id, xs, ys


def format_yolo_seg_line(cls_id, xs, ys):
    coords = []
    for x, y in zip(xs, ys):
        coords.append(f"{float(x):.6f}")
        coords.append(f"{float(y):.6f}")
    return f"{cls_id} " + " ".join(coords)


def polygon_to_mask(xs, ys, img_w, img_h):
    pts = np.stack([xs * img_w, ys * img_h], axis=1)
    pts = np.round(pts).astype(np.int32)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def bbox_from_mask(mask, padding=25):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x1 = x_min - padding
    y1 = y_min - padding
    x2 = x_max + padding
    y2 = y_max + padding

    return x1, y1, x2, y2


def clip_bbox(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_w - 1, int(x2))
    y2 = min(img_h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def clip01(arr):
    return np.clip(arr, 0.0, 1.0)


def remap_label_line(line: str):
    parsed = parse_yolo_seg_line(line)
    if parsed is None:
        return None

    old_cls_id, xs, ys = parsed
    new_cls_id = CLASS_REMAP.get(old_cls_id, None)

    if new_cls_id is None:
        return None

    return format_yolo_seg_line(new_cls_id, xs, ys)


def copy_and_remap_dataset(source_root: Path, output_root: Path):
    stats = {
        "kept": {0: 0, 1: 0, 2: 0},
        "removed": {k: 0 for k in OLD_CLASS_NAMES.keys()},
    }

    for split in ["train", "val"]:
        src_images = source_root / split / "images"
        src_labels = source_root / split / "labels"

        dst_images = output_root / split / "images"
        dst_labels = output_root / split / "labels"

        for p in src_images.iterdir():
            if p.is_file():
                shutil.copy2(p, dst_images / p.name)

        for lbl_path in src_labels.glob("*.txt"):
            out_lines = []

            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue

                parsed = parse_yolo_seg_line(line)
                if parsed is None:
                    continue

                old_cls_id, xs, ys = parsed
                new_cls_id = CLASS_REMAP.get(old_cls_id, None)

                if new_cls_id is None:
                    stats["removed"][old_cls_id] += 1
                    continue

                out_lines.append(format_yolo_seg_line(new_cls_id, xs, ys))
                stats["kept"][new_cls_id] += 1

            out_lbl_path = dst_labels / lbl_path.name
            out_lbl_path.write_text(
                ("\n".join(out_lines) + "\n") if out_lines else "",
                encoding="utf-8"
            )

    return stats


def save_yaml(output_root: Path):
    yaml_path = output_root / "new_LaRS.yaml"
    lines = [
        f"path: {output_root.as_posix()}",
        "",
        "train: train/images",
        "val: val/images",
        "",
        "names:",
    ]
    for k, v in CLASS_NAMES.items():
        lines.append(f"  {k}: {v}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def create_crop_from_polygon(image, xs, ys, padding):
    img_h, img_w = image.shape[:2]

    mask = polygon_to_mask(xs, ys, img_w, img_h)
    bbox = bbox_from_mask(mask, padding=padding)
    if bbox is None:
        return None

    clipped = clip_bbox(*bbox, img_w, img_h)
    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped
    crop = image[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]

    if crop_h <= 1 or crop_w <= 1:
        return None

    abs_xs = xs * img_w
    abs_ys = ys * img_h

    new_xs = (abs_xs - x1) / crop_w
    new_ys = (abs_ys - y1) / crop_h

    return crop, clip01(new_xs), clip01(new_ys)


def apply_hflip(img, xs, ys):
    img2 = cv2.flip(img, 1)
    xs2 = 1.0 - xs
    ys2 = ys.copy()
    return img2, clip01(xs2), clip01(ys2)


def apply_photometric(aug_name, img, xs, ys):
    if aug_name == "bright_up":
        img2 = cv2.convertScaleAbs(img, alpha=1.0, beta=18)
    elif aug_name == "bright_down":
        img2 = cv2.convertScaleAbs(img, alpha=1.0, beta=-18)
    elif aug_name == "contrast_up":
        img2 = cv2.convertScaleAbs(img, alpha=1.20, beta=0)
    elif aug_name == "blur3":
        img2 = cv2.GaussianBlur(img, (3, 3), 0)
    else:
        img2 = img.copy()
    return img2, xs.copy(), ys.copy()


def make_augmented_variant(image, xs, ys, aug_name, base_padding):
    if aug_name == "base":
        return create_crop_from_polygon(image, xs, ys, padding=base_padding)

    elif aug_name == "pad_plus_10":
        return create_crop_from_polygon(image, xs, ys, padding=base_padding + 10)

    elif aug_name == "pad_minus_10":
        p = max(0, base_padding - 10)
        return create_crop_from_polygon(image, xs, ys, padding=p)

    else:
        base = create_crop_from_polygon(image, xs, ys, padding=base_padding)
        if base is None:
            return None

        crop_img, crop_xs, crop_ys = base

        if aug_name == "hflip":
            return apply_hflip(crop_img, crop_xs, crop_ys)

        elif aug_name in {"bright_up", "bright_down", "contrast_up", "blur3"}:
            return apply_photometric(aug_name, crop_img, crop_xs, crop_ys)

        return crop_img, crop_xs, crop_ys


def save_crop_and_label(out_img_path, out_lbl_path, image, cls_id, xs, ys):
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_lbl_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(out_img_path), image)
    if not ok:
        return False

    line = format_yolo_seg_line(cls_id, xs, ys) + "\n"
    out_lbl_path.write_text(line, encoding="utf-8")
    return True


def augment_train_only(source_root: Path, output_root: Path):
    split = "train"
    images_dir = source_root / split / "images"
    labels_dir = source_root / split / "labels"

    out_images = output_root / split / "images"
    out_labels = output_root / split / "labels"

    created_per_class = {k: 0 for k in CLASS_NAMES.keys()}

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = find_image(images_dir, label_path.stem)
        if image_path is None:
            print(f"[WARN] Нет изображения для {label_path.name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Не удалось прочитать {image_path}")
            continue

        lines = [l.strip() for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            continue

        for inst_idx, line in enumerate(lines):
            parsed = parse_yolo_seg_line(line)
            if parsed is None:
                print(f"[WARN] Некорректная строка: {label_path.name}, instance {inst_idx}")
                continue

            old_cls_id, xs, ys = parsed
            new_cls_id = CLASS_REMAP.get(old_cls_id, None)

            if new_cls_id is None:
                continue

            n_aug = AUGS_PER_INSTANCE.get(new_cls_id, 0)
            if n_aug <= 0:
                continue

            for aug_idx in range(n_aug):
                aug_name = AUGMENTATION_POOL[aug_idx % len(AUGMENTATION_POOL)]

                result = make_augmented_variant(
                    image=image,
                    xs=xs,
                    ys=ys,
                    aug_name=aug_name,
                    base_padding=PADDING,
                )
                if result is None:
                    continue

                aug_img, aug_xs, aug_ys = result

                suffix = image_path.suffix.lower()
                out_img_name = f"{label_path.stem}_crop_new{new_cls_id}_{inst_idx}_aug{aug_idx}{suffix}"
                out_lbl_name = f"{label_path.stem}_crop_new{new_cls_id}_{inst_idx}_aug{aug_idx}.txt"

                out_img_path = out_images / out_img_name
                out_lbl_path = out_labels / out_lbl_name

                ok = save_crop_and_label(
                    out_img_path=out_img_path,
                    out_lbl_path=out_lbl_path,
                    image=aug_img,
                    cls_id=new_cls_id,
                    xs=aug_xs,
                    ys=aug_ys,
                )
                if ok:
                    created_per_class[new_cls_id] += 1

    return created_per_class


def count_instances_in_split_original(root: Path, split: str):
    labels_dir = root / split / "labels"
    counts = {k: 0 for k in OLD_CLASS_NAMES.keys()}

    for lbl in labels_dir.glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            parsed = parse_yolo_seg_line(line)
            if parsed is None:
                continue

            cls_id, _, _ = parsed
            counts[cls_id] = counts.get(cls_id, 0) + 1

    return counts


def count_instances_in_split_new(root: Path, split: str):
    labels_dir = root / split / "labels"
    counts = {k: 0 for k in CLASS_NAMES.keys()}

    for lbl in labels_dir.glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            parsed = parse_yolo_seg_line(line)
            if parsed is None:
                continue

            cls_id, _, _ = parsed
            counts[cls_id] = counts.get(cls_id, 0) + 1

    return counts


def main():
    print("SOURCE_DATASET:", SOURCE_DATASET)
    print("OUTPUT_DATASET:", OUTPUT_DATASET)

    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(
            f"Не найден SOURCE_DATASET: {SOURCE_DATASET}\n"
            f"Укажи правильный путь к instance_yolo."
        )

    ensure_structure(OUTPUT_DATASET)

    print("1) Считаю исходное число инстансов в train...")
    before_old = count_instances_in_split_original(SOURCE_DATASET, "train")
    for cls_id in sorted(OLD_CLASS_NAMES.keys()):
        print(f"  {cls_id} ({OLD_CLASS_NAMES[cls_id]}): {before_old.get(cls_id, 0)}")

    print("\n2) Копирую изображения и remap-лю labels в новый 3-классовый датасет...")
    remap_stats = copy_and_remap_dataset(SOURCE_DATASET, OUTPUT_DATASET)

    print("\nПосле remap в новом train+val датасете сохранено:")
    for cls_id in sorted(CLASS_NAMES.keys()):
        print(f"  {cls_id} ({CLASS_NAMES[cls_id]}): {remap_stats['kept'].get(cls_id, 0)}")

    print("\nУдалено объектов:")
    for cls_id in sorted(OLD_CLASS_NAMES.keys()):
        if remap_stats["removed"].get(cls_id, 0) > 0:
            print(f"  {cls_id} ({OLD_CLASS_NAMES[cls_id]}): {remap_stats['removed'][cls_id]}")

    print("\n3) Добавляю новые train-кропы только для buoy и other...")
    created = augment_train_only(SOURCE_DATASET, OUTPUT_DATASET)

    print("\nДобавлено новых train-примеров:")
    for cls_id in sorted(CLASS_NAMES.keys()):
        print(f"  {cls_id} ({CLASS_NAMES[cls_id]}): +{created.get(cls_id, 0)}")

    print("\n4) Считаю итоговые инстансы в новом train...")
    after_new_train = count_instances_in_split_new(OUTPUT_DATASET, "train")
    for cls_id in sorted(CLASS_NAMES.keys()):
        print(f"  {cls_id} ({CLASS_NAMES[cls_id]}): {after_new_train.get(cls_id, 0)}")

    print("\n5) Создаю YAML...")
    yaml_path = save_yaml(OUTPUT_DATASET)

    print("\nГотово.")
    print(f"Новый датасет: {OUTPUT_DATASET}")
    print(f"YAML: {yaml_path}")


if __name__ == "__main__":
    main()