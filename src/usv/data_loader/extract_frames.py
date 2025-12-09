"""
Script to extract frames from Left Camera and Center Phone.
Automatically fixes phone rotation issues.
"""
import cv2
from pathlib import Path
import tqdm

def extract_mixed_frames(
    input_dirs: list,
    output_dir: str = "data/interim/labeling_v3",
    interval_sec: int = 15
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
        for ext in ['*.avi', '*.mp4', '*.MOV']:
            video_files.extend(list(p.rglob(ext)))

    video_files.sort()
    print(f"Found {len(video_files)} videos in total.")

    global_idx = 0

    for vid_file in tqdm.tqdm(video_files, desc="Extracting"):
        cap = cv2.VideoCapture(str(vid_file))
        if not cap.isOpened(): continue

        # --- DETECTION LOGIC ---
        # Определяем, телефон это или нет, по имени папки или файла
        is_phone = "phone" in str(vid_file).lower() or "center" in str(vid_file).lower()

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30

        step = int(fps * interval_sec)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break

            # --- ROTATION FIX ---
            if is_phone:
                # Варианты: ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180
                # Если телефон снимал вертикально, обычно нужно CLOCKWISE
                try:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                except Exception as e:
                    print(f"Rotation failed: {e}")

            # Формируем имя с префиксом, чтобы различать источники
            # left_0001.jpg или phone_0002.jpg
            prefix = "phone" if is_phone else "left"
            fname = f"{prefix}_{global_idx:05d}.jpg"

            cv2.imwrite(str(output_path / fname), frame)
            global_idx += 1

        cap.release()

    print(f"Done. Saved {global_idx} images to {output_dir}")

if __name__ == "__main__":
    # Укажи папки, где лежат сырые видео
    folders = [
        "data/raw/left_cam",
        "data/raw/center_phone"
    ]

    extract_mixed_frames(
        input_dirs=folders,
        output_dir="data/interim/labeling_v3",
        interval_sec=20 # 1 кадр раз в 20 секунд
    )
