import os
import shutil
import random
from pathlib import Path

# =========================
# CONFIG
# =========================

# Корневая папка, где лежат исходные поддатасеты:
# our_dataset/
#   ultralytics_yolo_segmentation_1_0/
#   ultralytics_yolo_segmentation_1_0_02/
#   ...
SOURCE_ROOT = Path(r"E:\Education\4 course 2 semester\Diploma\panoptic_project\data\our_dataset1")

# Куда сохранить объединённый датасет
OUTPUT_ROOT = Path(r"E:\Education\4 course 2 semester\Diploma\panoptic_project\data\our_dataset")

# Доля валидации
VAL_RATIO = 0.1

# Сид для воспроизводимости
SEED = 42

# Расширения изображений
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Старые классы -> новые классы
# Исходные:
# 0: Pier
# 1: Bridge
# 2: Water
# 3: Sky
# 4: Land
# 5: Vessel
# 6: Buoy
# 7: LandingMark
# 8: BridgeLight
# 9: Other
# 10: Void
CLASS_MAP = {
    5: 0,  # Vessel
    6: 1,  # Buoy
    7: 2,  # LandingMark
    8: 3,  # BridgeLight
    9: 4,  # Other
}

NEW_NAMES = {
    0: "Vessel",
    1: "Buoy",
    2: "LandingMark",
    3: "BridgeLight",
    4: "Other",
}

YAML_PATH_VALUE = r"E:/Education/4 course 2 semester/Diploma/panoptic_project/data/our_dataset"

# Если True — изображения без оставшихся объектов не сохраняются
# Если False — такие изображения тоже попадут в датасет, но без label-файла
SKIP_EMPTY_IMAGES = True


# =========================
# HELPERS
# =========================

def find_subdatasets(source_root: Path):
    """
    Ищет подпапки, где есть:
      images/train
      labels/train
    """
    subdatasets = []
    for item in source_root.iterdir():
        if not item.is_dir():
            continue
        img_dir = item / "images" / "train"
        lbl_dir = item / "labels" / "train"
        if img_dir.exists() and lbl_dir.exists():
            subdatasets.append(item)
    return subdatasets


def find_image_for_label(label_path: Path, image_dir: Path):
    """
    Ищет изображение с тем же stem и любым допустимым расширением.
    """
    stem = label_path.stem
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def filter_and_remap_label(src_label_path: Path, dst_label_path: Path):
    """
    Читает YOLO-seg label, оставляет только нужные классы и переиндексирует их.
    Возвращает число оставшихся объектов.
    """
    kept_lines = []

    with open(src_label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 7:
            # Для segmentation должно быть минимум:
            # class x1 y1 x2 y2 x3 y3
            continue

        try:
            old_cls = int(float(parts[0]))
        except ValueError:
            continue

        if old_cls not in CLASS_MAP:
            continue

        new_cls = CLASS_MAP[old_cls]
        new_line = " ".join([str(new_cls)] + parts[1:])
        kept_lines.append(new_line)

    if kept_lines:
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(kept_lines) + "\n")
    elif dst_label_path.exists():
        dst_label_path.unlink()

    return len(kept_lines)


def safe_copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_yaml(yaml_path: Path):
    yaml_text = f"""path: {YAML_PATH_VALUE}

train: train/images
val: val/images

names:
  0: Vessel
  1: Buoy
  2: LandingMark
  3: BridgeLight
  4: Other
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)


# =========================
# MAIN
# =========================

def main():
    random.seed(SEED)

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"SOURCE_ROOT not found: {SOURCE_ROOT}")

    subdatasets = find_subdatasets(SOURCE_ROOT)
    if not subdatasets:
        raise RuntimeError("Не найдено ни одного поддатасета с images/train и labels/train")

    print("Найдены поддатасеты:")
    for ds in subdatasets:
        print(" -", ds.name)

    # Очищаем output, если уже существует
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    train_images_dir = OUTPUT_ROOT / "train" / "images"
    train_labels_dir = OUTPUT_ROOT / "train" / "labels"
    val_images_dir = OUTPUT_ROOT / "val" / "images"
    val_labels_dir = OUTPUT_ROOT / "val" / "labels"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    total_labels_found = 0
    total_kept_objects = 0
    total_skipped_no_image = 0
    total_empty_after_filter = 0

    # Сначала собираем все пары image-label
    for ds in subdatasets:
        img_dir = ds / "images" / "train"
        lbl_dir = ds / "labels" / "train"

        for label_file in lbl_dir.glob("*.txt"):
            total_labels_found += 1
            image_file = find_image_for_label(label_file, img_dir)

            if image_file is None:
                total_skipped_no_image += 1
                print(f"[WARN] Image not found for label: {label_file}")
                continue

            samples.append({
                "dataset_name": ds.name,
                "label_path": label_file,
                "image_path": image_file,
                "stem": label_file.stem,
            })

    if not samples:
        raise RuntimeError("Не найдено ни одной корректной пары image-label")

    random.shuffle(samples)

    val_count = int(len(samples) * VAL_RATIO)
    val_indices = set(range(val_count))

    counters = {}
    saved_images = 0
    saved_labels = 0

    for idx, sample in enumerate(samples):
        split = "val" if idx in val_indices else "train"

        ds_name = sample["dataset_name"]
        stem = sample["stem"]
        image_src = sample["image_path"]
        label_src = sample["label_path"]

        counters.setdefault(ds_name, 0)
        counters[ds_name] += 1

        unique_base = f"{ds_name}__{stem}_{counters[ds_name]:06d}"
        image_ext = image_src.suffix.lower()

        if split == "train":
            image_dst = train_images_dir / f"{unique_base}{image_ext}"
            label_dst = train_labels_dir / f"{unique_base}.txt"
        else:
            image_dst = val_images_dir / f"{unique_base}{image_ext}"
            label_dst = val_labels_dir / f"{unique_base}.txt"

        kept = filter_and_remap_label(label_src, label_dst)

        if kept == 0:
            total_empty_after_filter += 1
            if SKIP_EMPTY_IMAGES:
                if label_dst.exists():
                    label_dst.unlink()
                continue

        safe_copy_image(image_src, image_dst)
        saved_images += 1

        if kept > 0:
            saved_labels += 1
            total_kept_objects += kept

    write_yaml(OUTPUT_ROOT / "our_dataset.yaml")

    print("\n=== DONE ===")
    print(f"Всего label-файлов найдено: {total_labels_found}")
    print(f"Корректных sample-пар: {len(samples)}")
    print(f"Пропущено (нет изображения): {total_skipped_no_image}")
    print(f"Пустые после фильтрации: {total_empty_after_filter}")
    print(f"Сохранено изображений: {saved_images}")
    print(f"Сохранено label-файлов: {saved_labels}")
    print(f"Всего оставлено объектов: {total_kept_objects}")
    print(f"Результат: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()