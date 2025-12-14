"""
Super-Resolution Enhancement Script using OpenCV DNN.
Upscales Dashcam images to reduce pixelation and artifacts.
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_sr_model(model_name="EDSR", scale=4):
    """
    Initializes OpenCV DNN SuperRes object.
    """
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    model_path = Path(f"models/super_res/{model_name}_x{scale}.pb")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run download_sr_models.py first.")

    sr.readModel(str(model_path))
    sr.setModel(model_name.lower(), scale)
    return sr

def upscale_dataset(input_dir, output_dir, model="EDSR", scale=4):
    src_path = Path(input_dir)
    dst_path = Path(output_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    images = list(src_path.glob("*.jpg"))
    print(f"Upscaling {len(images)} images using {model} x{scale}...")
    print("Warning: EDSR is slow on CPU. Be patient.")

    # Инициализация модели
    sr = get_sr_model(model, scale)

    for img_file in tqdm(images):
        img = cv2.imread(str(img_file))
        if img is None: continue

        filename = img_file.name.lower()
        is_phone = "phone" in filename or "center" in filename

        # Логика:
        # Телефоны (4K) апскейлить НЕ НАДО (они и так огромные, лопнет память).
        # Камеры (Low Res) апскейлим.

        if is_phone:
            # Телефон просто копируем (или ресайзим вниз, если слишком большие)
            # Для единообразия можно сделать CLAHE
            # Но апскейл не делаем
            result = img
        else:
            # КАМЕРА: Делаем Super-Resolution
            # EDSR уберет пикселизацию
            result = sr.upsample(img)

        cv2.imwrite(str(dst_path / img_file.name), result)

    print(f"Done. Saved to {dst_path}")

if __name__ == "__main__":
    # Пробуем EDSR x4. Если будет слишком медленно/тяжело — поменяй на FSRCNN и scale=3
    # Вход: Оригинальные "labeling_v1" (НЕ enhanced, лучше апскейлить сырое)
    upscale_dataset(
        input_dir="data/interim/labeling_v4",
        output_dir="data/interim/labeling_upscaled_v4",
        model="FSRCNN", # Попробуй 'EDSR' (качество) или 'FSRCNN' (скорость)
        scale=3       # Для EDSR есть веса x4, для FSRCNN x3
    )
