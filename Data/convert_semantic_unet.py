import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

LARS_ROOT = Path(r"E:\Education\4 course 2 semester\Diploma\panoptic_project\data\LaRS")
OUT_ROOT = Path(r"E:\Education\4 course 2 semester\Diploma\panoptic_project\data\LaRS_UNet")

SPLITS = {
    "train": {
        "img_dir": LARS_ROOT / "image" / "train" / "images",
        "mask_dir": LARS_ROOT / "annotations" / "train" / "semantic_masks",
    },
    "val": {
        "img_dir": LARS_ROOT / "image" / "val" / "images",
        "mask_dir": LARS_ROOT / "annotations" / "val" / "semantic_masks",
    },
}

COLOR_TO_CLASS = {
    (247, 195, 37): 0,   # obstacle
    (41, 167, 224): 1,   # water
    (90, 75, 164): 2,    # sky
}

def list_files_by_stem(dir_path: Path):
    files = {}
    for p in dir_path.iterdir():
        if p.is_file():
            files[p.stem.lower()] = p
    return files

def rgb_mask_to_class_mask(mask_bgr: np.ndarray) -> np.ndarray:
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = mask_rgb.shape
    class_mask = np.full((h, w), 255, dtype=np.uint8)

    for color_rgb, class_id in COLOR_TO_CLASS.items():
        color = np.array(color_rgb, dtype=np.uint8)
        matches = np.all(mask_rgb == color, axis=-1)
        class_mask[matches] = class_id

    unknown = np.sum(class_mask == 255)
    if unknown > 0:
        print(f"Warning: {unknown} pixels were not mapped to any class.")

    return class_mask

def convert_split(split_name: str, img_dir: Path, mask_dir: Path):
    print(f"\n=== {split_name} ===")
    print("img_dir:", img_dir, "exists:", img_dir.exists())
    print("mask_dir:", mask_dir, "exists:", mask_dir.exists())

    out_img_dir = OUT_ROOT / split_name / "images"
    out_lbl_dir = OUT_ROOT / split_name / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = list_files_by_stem(img_dir)
    mask_files = list_files_by_stem(mask_dir)

    common = sorted(set(img_files.keys()) & set(mask_files.keys()))
    print(f"Found {len(common)} pairs in {split_name}")

    for stem in tqdm(common, desc=f"Processing {split_name}"):
        img_path = img_files[stem]
        mask_path = mask_files[stem]

        shutil.copy2(img_path, out_img_dir / f"{stem}{img_path.suffix.lower()}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Cannot read mask: {mask_path}")
            continue

        if mask.ndim == 2:
            class_mask = mask.astype(np.uint8)
        elif mask.ndim == 3:
            class_mask = rgb_mask_to_class_mask(mask)
        else:
            print(f"Unsupported mask shape: {mask.shape} for {mask_path}")
            continue

        cv2.imwrite(str(out_lbl_dir / f"{stem}.png"), class_mask)

def main():
    for split, cfg in SPLITS.items():
        convert_split(split, cfg["img_dir"], cfg["mask_dir"])

    print("\nDone.")
    print("Output dataset:", OUT_ROOT)

if __name__ == "__main__":
    main()