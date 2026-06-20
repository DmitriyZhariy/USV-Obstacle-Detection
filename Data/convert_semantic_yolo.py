import cv2
import numpy as np
import json
import os
from pathlib import Path

def mask_to_yolo_polygon(mask_path, output_path, image_width, image_height):
    """Конвертирует semantic mask в YOLO формат"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes != 0]  # Убираем фон
    
    with open(output_path, 'w') as f:
        for class_id in unique_classes:
            binary_mask = (mask == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                # Упрощаем полигон
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < 3:
                    continue
                
                # Нормализуем координаты
                normalized_points = []
                for point in approx:
                    x = point[0][0] / image_width
                    y = point[0][1] / image_height
                    normalized_points.extend([x, y])
                
                # Записываем в YOLO формат
                line = f"{class_id - 1} " + " ".join([f"{coord:.6f}" for coord in normalized_points])
                f.write(line + "\n")

def convert_lars_to_yolo(lars_root, output_root):
    """Конвертирует весь датасет LaRS в YOLO формат"""
    lars_root = Path(lars_root)
    output_root = Path(output_root)
    
    for split in ['train', 'val', 'test']:
        # Создаем директории
        (output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Пути к данным
        images_dir = lars_root / 'image' / split / 'images'
        masks_dir = lars_root / 'annotations' / split / 'semantic_masks'
        
        if not images_dir.exists():
            continue
        
        # Обрабатываем каждое изображение
        for img_path in images_dir.glob('*.jpg'):
            img_name = img_path.stem
            mask_path = masks_dir / f"{img_name}.png"
            
            if not mask_path.exists():
                continue
            
            # Копируем изображение
            img = cv2.imread(str(img_path))
            height, width = img.shape[:2]
            cv2.imwrite(str(output_root / 'images' / split / img_path.name), img)
            
            # Конвертируем маску
            output_label = output_root / 'labels' / split / f"{img_name}.txt"
            mask_to_yolo_polygon(mask_path, output_label, width, height)

# Использование
convert_lars_to_yolo(r"E:\Education\4 course 2 semester\Practice\panoptic_project\Diploma\Data\LaRS", "E:\Education\4 course 2 semester\Practice\panoptic_project\Diploma\Data\semantic_yolo")