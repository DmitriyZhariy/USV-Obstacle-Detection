import cv2
from pathlib import Path

input_video_path = r"D:\Study\4 course\USV-Obstacle-Detection\data\interim\choosed_clips_v5-1\video\right_MOVI0018_0002.mp4"
output_root_path = r"D:\Study\4 course\USV-Obstacle-Detection\data\interim\choosed_clips_v5-1\frames"

input_path = Path(input_video_path).resolve()
output_root = Path(output_root_path).resolve()

video_name = input_path.stem
output_dir = output_root / video_name
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(input_path))
frame_index = 1

while True:
    success, frame = cap.read()
    if not success:
        break

    cv2.imwrite(str(output_dir / f"{frame_index:04d}.jpg"), frame)
    frame_index += 1

cap.release()
print("Готово")