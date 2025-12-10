"""
Applies a static ignore mask to specified images in a dataset.

This script is designed to process a directory of frames, identify those
from a specific camera (e.g., a center-mounted phone), and apply a predefined
mask to them. This is useful for blacking out parts of the image that
should be ignored during model training, such as the ego-vehicle's bow.

Images not matching the criteria (e.g., from side dashcams) are copied
to the destination directory unmodified.
"""
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

def process_frames_with_mask(
    input_dir: str,
    output_dir: str,
    mask_path: str,
    camera_identifiers: list = ["phone", "center"]
):
    """
    Applies an ignore mask to images containing specific identifiers in their names.

    Args:
        input_dir (str): Path to the directory with source frames.
        output_dir (str): Path to the directory where processed frames will be saved.
        mask_path (str): Path to the ignore mask image (black area = ignore).
        camera_identifiers (list): A list of substrings to identify target frames.
    """
    src_path = Path(input_dir)
    dst_path = Path(output_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    # 1. Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Mask file not found at '{mask_path}'.")
        print("Please create a mask image where the area to be ignored is black.")
        return

    # 2. Find all images
    image_files = list(src_path.glob("*.jpg")) + list(src_path.glob("*.png"))
    if not image_files:
        print(f"Warning: No images found in '{src_path}'.")
        return

    print(f"Processing {len(image_files)} images from '{src_path}'...")

    # 3. Process each image
    mask_resized = None # Cache for resized mask

    for img_file in tqdm(image_files):
        filename_lower = img_file.name.lower()

        # Check if any identifier is in the filename
        should_apply_mask = any(id in filename_lower for id in camera_identifiers)

        if should_apply_mask:
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Warning: Could not read image {img_file.name}, skipping.")
                continue

            h, w = img.shape[:2]

            # 4. Resize mask if necessary (and cache it)
            if mask_resized is None or mask_resized.shape[0] != h or mask_resized.shape[1] != w:
                mask_resized = cv2.resize(mask, (w, h))

            # 5. Apply mask (bitwise_and makes black areas in mask black in output)
            result_img = cv2.bitwise_and(img, img, mask=mask_resized)

            # Save the masked image
            cv2.imwrite(str(dst_path / img_file.name), result_img)
        else:
            # If no identifier matches, just copy the file
            shutil.copy(img_file, dst_path / img_file.name)

    print("-" * 30)
    print("Processing complete.")
    print(f"All frames have been saved to: '{dst_path}'")
    print("You can now use this directory for labeling.")


if __name__ == "__main__":
    # --- Configuration ---
    # Create this mask manually. The part of your boat should be BLACK,
    # and the rest of the scene should be WHITE.
    MASK_FILE = "data/interim/phone_ignore_mask.png"

    # Make sure the mask file exists before running
    if not Path(MASK_FILE).exists():
        print(f"ERROR: Mask file not found at '{MASK_FILE}'")
        print("Please create it first. It should be a PNG file.")
        print("The area to HIDE (your boat) must be BLACK.")
        print("The area to KEEP must be WHITE.")
    else:
        process_frames_with_mask(
            input_dir="data/interim/labeling_upscaled",
            output_dir="data/interim/labeling_upscaled_masked",
            mask_path=MASK_FILE
        )
