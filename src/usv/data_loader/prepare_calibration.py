import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
import uuid

def prepare_calibration_set(source_dir, output_dir, annotators):
    src_path = Path(source_dir)
    dst_path = Path(output_dir)

    # Полная очистка
    if dst_path.exists():
        shutil.rmtree(dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)

    images = list(src_path.glob("*.[jJ][pP]*[gG]")) + list(src_path.glob("*.png"))

    print(f"Applying resolution change (padding) for: {annotators}")

    for img_path in tqdm(images):
        original_img = cv2.imread(str(img_path))
        if original_img is None:
            continue

        for i, user in enumerate(annotators):
            # --- THE FIX: CHANGE IMAGE HEIGHT ---
            # Мы добавляем черную полоску снизу.
            # User 0: +0 px
            # User 1: +1 px
            # ...
            # Это меняет разрешение (Resolution) картинки.
            # Roboflow не может считать дубликатами картинки разного размера.

            pad_height = i  # 0, 1, 2, 3... pixels

            if pad_height > 0:
                # (top, bottom, left, right), borderType, color
                unique_img = cv2.copyMakeBorder(
                    original_img,
                    0, pad_height, 0, 0,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0] # Черный цвет
                )
            else:
                unique_img = original_img.copy()

            # --- ДОПОЛНИТЕЛЬНАЯ ЗАЩИТА: СЛУЧАЙНОЕ ИМЯ ---
            # Добавляем случайный ID, чтобы имена файлов точно не совпали
            random_id = str(uuid.uuid4())[:8]
            new_name = f"{user}__{img_path.stem}__{random_id}.png"

            cv2.imwrite(str(dst_path / new_name), unique_img)

    print(f"Done! Files saved to: {dst_path}")
    print("Each user now has images of slightly different heights.")


if __name__ == "__main__":
    # Список никнеймов твоих разметчиков
    ANNOTATORS = ["Dmitry", "Evgeny", "Karina", "Nikita"]

    prepare_calibration_set(
        source_dir="data/interim/labeling_for_markup_test",
        output_dir="data/interim/calibration_ready",
        annotators=ANNOTATORS
    )
