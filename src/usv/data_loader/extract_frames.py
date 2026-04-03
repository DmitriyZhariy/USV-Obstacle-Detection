"""
Script to extract frames from Left Camera and Center Phone.
Automatically fixes phone rotation issues.
"""
import cv2
from pathlib import Path
import tqdm

def extract_mixed_frames(
    input_dirs: list,
    output_dir: str = "data/interim/labeling_v5",
    interval_sec: int = 15,
    rotate_center: bool = True  # Опция для включения/выключения поворота
):
    """
    Args:
        input_dirs: List of folders (e.g. ["data/raw/left_cam", "data/raw/center_phone"])
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Собираем все видео из всех указанных папок
    video_files = []
    for d in input_dirs:
        p = Path(d)
        for ext in ['*.avi', '*.mp4', '*.MOV', '*.mkv']: # Добавил mkv на всякий случай
            video_files.extend(list(p.rglob(ext)))

    video_files.sort()
    print(f"Found {len(video_files)} videos in total.")

    global_idx = 0

    for vid_file in tqdm.tqdm(video_files, desc="Extracting"):
        cap = cv2.VideoCapture(str(vid_file))
        if not cap.isOpened():
            print(f"Error opening {vid_file}")
            continue

        # --- ИСПРАВЛЕННАЯ ЛОГИКА ОПРЕДЕЛЕНИЯ ИСТОЧНИКА ---
        # Проверяем полный путь (строку), чтобы найти название папки
        path_str = str(vid_file).lower()

        if "phone" in path_str or "center" in path_str:
            prefix = "center"
        elif "right" in path_str:
            prefix = "right"
        else:
            prefix = "left"

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30

        # Защита от слишком большого интервала
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        step = int(fps * interval_sec)
        if step == 0: step = 1 # Чтобы не было бесконечного цикла или ошибки

        # Проходим по кадрам
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                # Иногда set() срабатывает, но read() в конце видео может вернуть False
                break

            # --- ROTATION FIX ---
            # Применяем только к center и если включена опция
            if prefix == 'center' and rotate_center:
                try:
                    # ROTATE_90_CLOCKWISE подходит, если видео снято вертикально,
                    # но OpenCV открывает его "лежа" на левом боку.
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                except Exception as e:
                    print(f"Rotation failed on {vid_file}: {e}")

            # Формируем имя
            fname = f"{prefix}_{global_idx:05d}.jpg"

            # Сохраняем
            cv2.imwrite(str(output_path / fname), frame)
            global_idx += 1

        cap.release()

    print(f"Done. Saved {global_idx} images to {output_dir}")

if __name__ == "__main__":
    # Укажи папки, где лежат сырые видео
    folders = [
        "data/raw/left_cam",
        "data/raw/right_cam",
        "data/raw/center_phone"
    ]

    extract_mixed_frames(
        input_dirs=folders,
        output_dir="data/interim/labeling_v5",
        interval_sec=5,
        rotate_center=True
    )
