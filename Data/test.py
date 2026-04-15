import cv2
import numpy as np
import os

# --- пути ---
mask_path = r"E:\Education\4 course 2 semester\Practice\panoptic_project\Data\LaRS_fusion\semantic\train\masks\davimar_seq_01_00017.png"
label_path = r"E:\Education\4 course 2 semester\Practice\panoptic_project\Data\LaRS_fusion\instance_yolo\train\labels\davimar_seq_01_00017.txt"

# =========================
# 🔹 SEMANTIC CHECK
# =========================
mask = cv2.imread(mask_path, 0)

if mask is None:
    print("Semantic mask не найдена")
else:
    unique_classes = np.unique(mask)
    print("Semantic classes:", unique_classes)
    print("Semantic разметка выглядит корректной")



# =========================
# 🔹 INSTANCE CHECK (YOLO)
# =========================
if not os.path.exists(label_path):
    print("Instance label не найден")
else:
    with open(label_path, 'r') as f:
        lines = f.readlines()

    if len(lines) == 0:
        print("⚠️ Instance файл пустой")
    else:
        classes = set()
        errors = False

        for i, line in enumerate(lines):
            parts = line.strip().split()

            if len(parts) < 3:
                print(f"Строка {i}: слишком мало данных")
                errors = True
                continue

            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            classes.add(class_id)

            # проверка чётности координат
            if len(coords) % 2 != 0:
                print(f"Строка {i}: нечетное число координат")
                errors = True

            coords = np.array(coords)

            # проверка диапазона
            if np.any(coords < 0) or np.any(coords > 1):
                print(f"Строка {i}: координаты вне [0,1]")
                errors = True

        print("Instance classes:", sorted(classes))
        print("Количество объектов:", len(lines))

        if not errors:
            print("Instance разметка выглядит корректной")