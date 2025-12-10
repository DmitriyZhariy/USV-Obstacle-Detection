"""
Image Enhancement Script v3.
Pipeline:
- Phone: CLAHE (it worked well previously)
- Dashcam: Gamma Correction (Brightening) + Bilateral Filter (Edge-preserving smoothing)
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def apply_gamma(img, gamma=1.2):
    """
    Non-linear brightness adjustment.
    gamma > 1.0 makes shadows brighter without washing out highlights.
    Great for low-contrast dashcam footage.
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_bilateral(img):
    """
    Bilateral Filter replaces a pixel with an average of similar nearby pixels.
    It smooths noise but KEEPS EDGES sharp.
    d=9: Diameter of pixel neighborhood.
    sigmaColor=75: How different colors must be to NOT get mixed (higher = more blur).
    sigmaSpace=75: How far pixels can be to influence each other.
    """
    return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

def apply_clahe(img, clip_limit=2.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def enhance_dataset(input_dir, output_dir):
    src_path = Path(input_dir)
    dst_path = Path(output_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    images = list(src_path.glob("*.jpg"))
    print(f"Enhancing {len(images)} images from {input_dir}...")

    for img_file in tqdm(images):
        img = cv2.imread(str(img_file))
        if img is None: continue

        filename = img_file.name.lower()
        is_phone = "phone" in filename or "center" in filename

        if is_phone:
            # ТЕЛЕФОН: Оставляем CLAHE, раз он работал хорошо
            # Можно добавить легкий билатерал, чтобы убрать зерно
            img_clahe = apply_clahe(img, clip_limit=2.0)
            result = apply_bilateral(img_clahe)

        else:
            # КАМЕРА:
            # 1. Гамма-коррекция вместо CLAHE (меньше шума)
            # gamma=1.3 высветлит темную воду/берег
            img_gamma = apply_gamma(img, gamma=1.3)

            # 2. Билатеральный фильтр вместо Denoise
            # Он сохранит край берега, но уберет "квадратики" сжатия на небе/воде
            result = apply_bilateral(img_gamma)

        cv2.imwrite(str(dst_path / img_file.name), result)

    print(f"Done. Enhanced images saved to {output_dir}")

if __name__ == "__main__":
    enhance_dataset(
        input_dir="data/interim/labeling_v3",
        output_dir="data/interim/labeling_enhanced_v3"
    )
