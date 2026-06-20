import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


# =========================
# НАСТРОЙКИ
# =========================

ROOT = r"E:\Education\4 course 2 semester\Practice\panoptic_project\Data\River"
OUT_ROOT = r"E:\Education\4 course 2 semester\Practice\panoptic_project\Data\River_YOLO"

TRAIN_RATIO = 0.8
SEED = 42

# Оставляем только things-классы
THING_CLASSES = [
    "Vessel",
    "Buoy",
    "LandingMark",
    "BridgeLight",
    "Pier",
]

class_to_id = {name: i for i, name in enumerate(THING_CLASSES)}


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_image_list(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    result = []
    if not os.path.isdir(folder):
        return result

    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and Path(name).suffix.lower() in exts:
            result.append(p)
    return sorted(result)


def collect_image_dirs(images_root):
    if not os.path.isdir(images_root):
        return []

    subdirs = [
        os.path.join(images_root, d)
        for d in sorted(os.listdir(images_root))
        if os.path.isdir(os.path.join(images_root, d))
    ]

    if subdirs:
        return subdirs

    return [images_root]


def collect_all_images(image_dirs):
    all_images = []
    for d in image_dirs:
        all_images.extend(get_image_list(d))
    return sorted(all_images)


def read_image_size(image_path):
    if Image is None:
        raise RuntimeError(
            "Pillow не установлен, а размер кадра нужно читать из изображения. "
            "Установите: pip install pillow"
        )
    with Image.open(image_path) as img:
        return img.size


def parse_points(points_str, width, height):
    pts = []
    for p in points_str.split(";"):
        p = p.strip()
        if not p:
            continue

        xy = p.split(",")
        if len(xy) != 2:
            continue

        x, y = map(float, xy)

        x = max(0.0, min(1.0, x / width))
        y = max(0.0, min(1.0, y / height))
        pts.append((x, y))

    return pts


def find_image_by_name(image_dirs, image_name):
    for d in image_dirs:
        candidate = os.path.join(d, image_name)
        if os.path.exists(candidate):
            return candidate
    return None


def make_unique_output_name(subfolder_name, base_name):
    return f"{subfolder_name}__{base_name}"


# =========================
# ПАРСИНГ CVAT XML
# =========================

def parse_annotations_xml(xml_path, sample_image_path=None):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = None
    height = None

    # 1) Пробуем стандартный путь meta/original_size
    width_el = root.find("./meta/original_size/width")
    height_el = root.find("./meta/original_size/height")

    if width_el is not None and height_el is not None:
        if width_el.text is not None and height_el.text is not None:
            width = int(float(width_el.text))
            height = int(float(height_el.text))

    # 2) Fallback: первый <image width="..." height="...">
    if width is None or height is None:
        first_image = root.find("image")
        if first_image is not None:
            w_attr = first_image.attrib.get("width")
            h_attr = first_image.attrib.get("height")
            if w_attr is not None and h_attr is not None:
                width = int(float(w_attr))
                height = int(float(h_attr))

    # 3) Fallback: взять размер из реального файла изображения
    if (width is None or height is None) and sample_image_path is not None:
        width, height = read_image_size(sample_image_path)

    if width is None or height is None:
        raise ValueError(
            f"Не удалось определить размер кадра в XML: {xml_path}"
        )

    frames = {}
    frame_to_name = {}

    # В вашем XML из paste.txt используется image-based annotation:
    # <image id="..." name="..." width="..." height="..."> <polygon .../>
    for image_el in root.findall("image"):
        frame_id = image_el.attrib.get("id")
        image_name = image_el.attrib.get("name")

        if frame_id is None:
            continue

        frame_id = int(frame_id)
        if image_name is not None:
            frame_to_name[frame_id] = image_name

        img_w = width
        img_h = height

        # Если у конкретного image свои width/height — лучше брать их
        if image_el.attrib.get("width") is not None and image_el.attrib.get("height") is not None:
            img_w = int(float(image_el.attrib["width"]))
            img_h = int(float(image_el.attrib["height"]))

        objects = []

        for poly in image_el.findall("polygon"):
            label = poly.attrib.get("label", "").strip()

            if label not in class_to_id:
                continue

            points_str = poly.attrib.get("points", "").strip()
            if not points_str:
                continue

            pts = parse_points(points_str, img_w, img_h)
            if len(pts) < 3:
                continue

            cid = class_to_id[label]
            objects.append((cid, pts))

        if objects:
            frames[frame_id] = objects

    # Дополнительно поддержка track-based XML, если вдруг попадётся другой экспорт:
    # <track label="..."><polygon frame="..." points="..."/></track>
    for track in root.findall("track"):
        label = track.attrib.get("label", "").strip()
        if label not in class_to_id:
            continue

        cid = class_to_id[label]

        for poly in track.findall("polygon"):
            outside = poly.attrib.get("outside", "0")
            if outside == "1":
                continue

            frame_str = poly.attrib.get("frame")
            points_str = poly.attrib.get("points", "").strip()

            if frame_str is None or not points_str:
                continue

            frame_id = int(frame_str)
            pts = parse_points(points_str, width, height)
            if len(pts) < 3:
                continue

            frames.setdefault(frame_id, []).append((cid, pts))

    return frames, frame_to_name, width, height


# =========================
# КОНВЕРТАЦИЯ ОДНОЙ ПАПКИ
# =========================

def convert_one_sequence(subdir, temp_images_dir, temp_labels_dir):
    sub_name = os.path.basename(subdir.rstrip("\\/"))
    xml_path = os.path.join(subdir, "annotations.xml")

    if not os.path.exists(xml_path):
        print(f"[SKIP] Нет annotations.xml: {subdir}")
        return 0

    images_root = os.path.join(subdir, "images")
    image_dirs = collect_image_dirs(images_root)
    all_images = collect_all_images(image_dirs)

    if not all_images:
        print(f"[SKIP] Не найдены изображения: {images_root}")
        return 0

    frames, frame_to_name, width, height = parse_annotations_xml(
        xml_path,
        sample_image_path=all_images[0]
    )

    print(f"[INFO] {sub_name}: frames with objects = {len(frames)}, size = {width}x{height}")

    saved = 0

    for frame_id, objs in frames.items():
        img_path = None

        # Предпочтительно брать точное имя кадра из XML
        if frame_to_name:
            img_name = frame_to_name.get(frame_id)
            if img_name is None:
                continue
            img_path = find_image_by_name(image_dirs, img_name)
        else:
            # fallback по индексу кадра
            if 0 <= frame_id < len(all_images):
                img_path = all_images[frame_id]

        if img_path is None or not os.path.exists(img_path):
            print(f"[WARN] Изображение не найдено для frame {frame_id} в {sub_name}")
            continue

        base_name = os.path.basename(img_path)
        stem = os.path.splitext(base_name)[0]
        ext = os.path.splitext(base_name)[1].lower()

        # Чтобы файлы из разных подпапок не перетирали друг друга
        out_base = make_unique_output_name(sub_name, stem)
        out_img_name = out_base + ext
        out_lbl_name = out_base + ".txt"

        out_img_path = os.path.join(temp_images_dir, out_img_name)
        out_lbl_path = os.path.join(temp_labels_dir, out_lbl_name)

        shutil.copy2(img_path, out_img_path)

        with open(out_lbl_path, "w", encoding="utf-8") as f:
            for cid, pts in objs:
                coords = []
                for x, y in pts:
                    coords.append(f"{x:.6f}")
                    coords.append(f"{y:.6f}")
                line = f"{cid} " + " ".join(coords)
                f.write(line + "\n")

        saved += 1

    return saved


# =========================
# SPLIT TRAIN / VAL
# =========================

def split_dataset(temp_images_dir, temp_labels_dir, out_root, train_ratio=0.8, seed=42):
    image_files = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    for name in os.listdir(temp_images_dir):
        p = os.path.join(temp_images_dir, name)
        if os.path.isfile(p) and Path(name).suffix.lower() in valid_exts:
            stem = os.path.splitext(name)[0]
            lbl = os.path.join(temp_labels_dir, stem + ".txt")
            if os.path.exists(lbl):
                image_files.append(name)

    image_files = sorted(image_files)
    random.Random(seed).shuffle(image_files)

    n_total = len(image_files)
    n_train = int(n_total * train_ratio)

    train_files = image_files[:n_train]
    val_files = image_files[n_train:]

    for split in ["train", "val"]:
        ensure_dir(os.path.join(out_root, "images", split))
        ensure_dir(os.path.join(out_root, "labels", split))

    for split_name, files in [("train", train_files), ("val", val_files)]:
        for img_name in files:
            stem = os.path.splitext(img_name)[0]
            lbl_name = stem + ".txt"

            src_img = os.path.join(temp_images_dir, img_name)
            src_lbl = os.path.join(temp_labels_dir, lbl_name)

            dst_img = os.path.join(out_root, "images", split_name, img_name)
            dst_lbl = os.path.join(out_root, "labels", split_name, lbl_name)

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

    return len(train_files), len(val_files)


def write_data_yaml(out_root):
    yaml_path = os.path.join(out_root, "data.yaml")

    lines = [
        f"path: {out_root.replace(os.sep, '/')}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(THING_CLASSES)}",
        "names:",
    ]

    for i, name in enumerate(THING_CLASSES):
        lines.append(f"  {i}: {name}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return yaml_path


# =========================
# MAIN
# =========================

def main():
    random.seed(SEED)

    ensure_dir(OUT_ROOT)

    temp_images_dir = os.path.join(OUT_ROOT, "_temp_all_images")
    temp_labels_dir = os.path.join(OUT_ROOT, "_temp_all_labels")

    if os.path.exists(temp_images_dir):
        shutil.rmtree(temp_images_dir)
    if os.path.exists(temp_labels_dir):
        shutil.rmtree(temp_labels_dir)

    ensure_dir(temp_images_dir)
    ensure_dir(temp_labels_dir)

    total_saved = 0

    subdirs = [
        os.path.join(ROOT, d)
        for d in sorted(os.listdir(ROOT))
        if os.path.isdir(os.path.join(ROOT, d))
    ]

    if not subdirs:
        print(f"Не найдены подпапки в ROOT: {ROOT}")
        return

    for subdir in subdirs:
        try:
            count = convert_one_sequence(subdir, temp_images_dir, temp_labels_dir)
            total_saved += count
        except Exception as e:
            print(f"[ERROR] Ошибка в {subdir}: {e}")

    if total_saved == 0:
        print("Не удалось сохранить ни одного примера.")
        return

    train_count, val_count = split_dataset(
        temp_images_dir=temp_images_dir,
        temp_labels_dir=temp_labels_dir,
        out_root=OUT_ROOT,
        train_ratio=TRAIN_RATIO,
        seed=SEED
    )

    yaml_path = write_data_yaml(OUT_ROOT)

    # Удаляем временные папки
    shutil.rmtree(temp_images_dir, ignore_errors=True)
    shutil.rmtree(temp_labels_dir, ignore_errors=True)

    print("\nГотово.")
    print(f"Всего объектов/кадров сохранено: {total_saved}")
    print(f"Train: {train_count}")
    print(f"Val:   {val_count}")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()