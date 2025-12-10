"""
Inference Script with Phone Support.
Runs the trained YOLOv8 model on a video file.
Supports rotation and resizing for high-res phone videos.
"""
import cv2
from ultralytics import YOLO
from pathlib import Path

def run_inference(
    video_path: str,
    model_path: str,
    output_path: str = "reports/demo_phone.mp4",
    conf_threshold: float = 0.2,
    rotate_code: int = None,    # cv2.ROTATE_90_CLOCKWISE и т.д.
    target_width: int = None    # Например, 1080. Если None - оригинальный размер.
):
    # 1. Load Model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # --- SETUP DIMENTIONS (READ 1st FRAME) ---
    # Нам нужно прочитать один кадр, чтобы понять итоговый размер
    # после поворота и ресайза, чтобы правильно настроить VideoWriter.
    ret, sample_frame = cap.read()
    if not ret:
        print("Video is empty.")
        return

    # Логика трансформации (вынесена, чтобы применять к каждому кадру)
    def process_frame(img):
        if rotate_code is not None:
            img = cv2.rotate(img, rotate_code)

        if target_width is not None:
            h, w = img.shape[:2]
            scale = target_width / w
            new_h = int(h * scale)
            img = cv2.resize(img, (target_width, new_h))
        return img

    # Проверяем размеры на первом кадре
    processed_sample = process_frame(sample_frame)
    out_h, out_w = processed_sample.shape[:2]

    # Сбрасываем видео в начало
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 3. Setup Output Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (out_w, out_h))

    print(f"Output resolution: {out_w}x{out_h}")
    print(f"Processing video... (Press 'q' to stop early)")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- PREPROCESSING ---
        frame = process_frame(frame)

        # --- INFERENCE ---
        # conf=0.4: порог уверенности
        # retina_masks=True: более качественные маски (чуть медленнее)
        results = model.predict(frame, conf=conf_threshold, verbose=False, retina_masks=True)

        # Визуализация
        annotated_frame = results[0].plot()

        # Запись
        out.write(annotated_frame)

        # Отображение
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    # --- НАСТРОЙКИ ДЛЯ ТЕЛЕФОНА ---

    # 1. Выбери видео
    video = "data/raw/center_phone/VID_20251027_151559.mp4"

    # 2. Поворот (обычно для телефона нужен CLOCKWISE)
    # Если лежит на боку влево -> используй cv2.ROTATE_90_CLOCKWISE
    # Если вверх ногами -> cv2.ROTATE_90_COUNTERCLOCKWISE
    # Если всё ок -> None
    rotation = cv2.ROTATE_90_CLOCKWISE

    # 3. Масштаб (чтобы влезло в экран и быстрее считалось)
    # 720 - оптимально для теста, 1080 - для качества
    width = 720

    run_inference(
        video_path=video,
        model_path="reports/models/mastr_baseline/weights/best.pt",
        output_path="reports/videos/phone_inference_test.mp4",
        rotate_code=rotation,
        target_width=width
    )
