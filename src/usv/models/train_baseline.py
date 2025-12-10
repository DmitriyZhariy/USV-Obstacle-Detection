"""
Script to train the Baseline model on MaSTr1325 dataset.
"""
from ultralytics import YOLO
from pathlib import Path
import os

def train_baseline():
    # 1. Setup paths
    # YOLO требует абсолютный путь к yaml, чтобы не запутаться
    project_root = Path.cwd()
    dataset_yaml = project_root / "data/processed/mastr_yolo/dataset.yaml"

    if not dataset_yaml.exists():
        print(f"Error: Dataset not found at {dataset_yaml}")
        return

    # 2. Load Model
    # 'yolov8s-seg.pt' will download automatically
    model = YOLO('yolov8s-seg.pt')

    # 3. Train
    print("Starting training on MaSTr1325...")
    results = model.train(
        data=str(dataset_yaml),
        project="reports/models", # Куда сохранять логи и веса
        name="mastr_baseline",    # Имя эксперимента
        epochs=50,                # 50 эпох хватит для бейзлайна
        imgsz=640,                # Размер картинки (стандарт)
        batch=4,                 # Если не влезет в память, уменьши до 8 или 4
        patience=10,              # Остановка, если нет улучшений 10 эпох
        device='cpu',                 # 0 для GPU, 'cpu' для процессора (медленно!)
        workers=4,
        exist_ok=True             # Перезаписывать эксперимент с тем же именем
    )

    print("Training finished.")
    print(f"Best model saved at: reports/models/mastr_baseline/weights/best.pt")

if __name__ == "__main__":
    train_baseline()
