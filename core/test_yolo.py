from ultralytics import YOLO
import cv2
import numpy as np
import os

# =========================
# УКАЖИ СВОИ ПУТИ ЗДЕСЬ
# =========================
MODEL_PATH = r"E:\Education\4 course 2 semester\Diploma\panoptic_project\runs\instance_segmentation\yolo\medium\best.pt"
IMAGE_PATH = r"E:\Education\4 course 2 semester\Diploma\panoptic_project\data\LaRS_fusion\instance_yolo\val\images\smd_seq01_00570.jpg"
OUTPUT_PATH = r"E:\Education\4 course 2 semester\Diploma\panoptic_project\metrics\instance_segmentation\yolo\yolo26m_our\output_segmented.png"

# Цвет маски в BGR (например, красный)
MASK_COLOR = (0, 0, 255)

# Прозрачность маски:
# 0.0 = полностью прозрачная
# 1.0 = полностью непрозрачная
ALPHA = 0.4


def main():
    # Загружаем модель
    model = YOLO(MODEL_PATH)

    # Читаем изображение
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {IMAGE_PATH}")

    # Инференс
    results = model.predict(source=IMAGE_PATH, save=False, conf=0.25)

    # Копия изображения для наложения масок
    output = image.copy()

    for result in results:
        if result.masks is None:
            print("Маски не найдены.")
            continue

        # result.masks.data -> tensor [N, H, W]
        masks = result.masks.data.cpu().numpy()

        for mask in masks:
            # Преобразуем маску в бинарный формат
            mask = (mask > 0.5).astype(np.uint8)

            # Если размер маски не совпадает с размером изображения — подгоним
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # Создаем цветной слой для маски
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            colored_mask[mask == 1] = MASK_COLOR

            # Накладываем полупрозрачную маску только в области объекта
            output = np.where(
                colored_mask > 0,
                cv2.addWeighted(output, 1 - ALPHA, colored_mask, ALPHA, 0),
                output
            )

    # Сохраняем результат
    cv2.imwrite(OUTPUT_PATH, output)
    print(f"Готово. Результат сохранен: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()