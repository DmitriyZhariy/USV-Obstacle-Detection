"""
Converter for MaSTr1325 dataset to YOLOv8 Segmentation format.
MaSTr masks usually: 0=Obstacle, 1=Water, 2=Sky, 4=Ignore.
We will map them to YOLO classes:
0: Water
1: Obstacle (Anything that is not water and not sky)
"""
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import random

def get_yolo_polygon(binary_mask):
    """
    Finds contours in a binary mask and normalizes them (0..1).
    Returns list of strings ready for YOLO txt file.
    """
    H, W = binary_mask.shape
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50: # Filter tiny noise
            continue

        # Simplify contour to reduce file size and training time
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3: continue

        # Normalize coordinates (x/W, y/H)
        coords = []
        for point in approx:
            x, y = point[0]
            coords.append(f"{x/W:.6f} {y/H:.6f}")

        polygons.append(coords)

    return polygons

def convert_mastr_to_yolo(input_dir, output_dir, split_ratio=0.8):
    root = Path(input_dir)
    images_dir = root / "images"
    masks_dir = root / "masks"

    # Setup YOLO structure
    save_path = Path(output_dir)
    for split in ['train', 'val']:
        (save_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (save_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    files = list(images_dir.glob("*.jpg"))
    random.shuffle(files)

    # YOLO Class Mapping
    # Мы хотим: 0 = Water, 1 = Obstacle
    # В MaSTr: 1 = Water, (0, 4) = Obstacle/Ignore, 2 = Sky (Ignore for obstacles)

    print(f"Found {len(files)} images. Starting conversion...")

    for i, img_path in tqdm(enumerate(files), total=len(files)):
        # Determine split
        split = 'train' if i < len(files) * split_ratio else 'val'

        # Read Mask
        mask_name = img_path.name.replace(".jpg", "m.png") # MaSTr masks usually end with 'm.png'
        mask_path = masks_dir / mask_name

        if not mask_path.exists():
            print(f"Mask not found for {img_path.name}, skipping.")
            continue

        # Load mask (Grayscale)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # --- CLASS 0: WATER ---
        # In MaSTr, Water is usually value 1
        water_mask = np.where(mask == 1, 255, 0).astype(np.uint8)
        water_polys = get_yolo_polygon(water_mask)

        # --- CLASS 1: OBSTACLE ---
        # In MaSTr, Obstacle is 0.
        # CAUTION: We should exclude SKY (value 2) and IGNORE (value 4) from obstacles.
        obstacle_mask = np.where(mask == 0, 255, 0).astype(np.uint8)
        obstacle_polys = get_yolo_polygon(obstacle_mask)

        # Prepare Label Content
        label_lines = []

        # Add Water (Class 0)
        for poly in water_polys:
            label_lines.append(f"0 {' '.join(poly)}")

        # Add Obstacle (Class 1)
        for poly in obstacle_polys:
            label_lines.append(f"1 {' '.join(poly)}")

        # Write Files
        if label_lines: # Only save if we have labels
            # Copy Image
            shutil.copy(img_path, save_path / split / 'images' / img_path.name)

            # Save Label txt
            txt_name = img_path.stem + ".txt"
            with open(save_path / split / 'labels' / txt_name, "w") as f:
                f.write("\n".join(label_lines))

    print("Conversion Finished!")

    # Create dataset.yaml for YOLO
    yaml_content = f"""
path: {save_path.absolute()}
train: train/images
val: val/images

names:
  0: water
  1: obstacle
"""
    with open(save_path / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    print(f"Created dataset.yaml at {save_path}")

if __name__ == "__main__":
    # Убедись, что пути верные
    convert_mastr_to_yolo(
        input_dir="data\external\MaSTr1325",
        output_dir="data/processed/mastr_yolo"
    )
