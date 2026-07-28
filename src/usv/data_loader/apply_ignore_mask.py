"""
Applies a static ignore mask to images selected by camera identifier.

Frames whose filename contains a configured identifier are masked. Other frames
are copied to the output directory unchanged. The expected use case is masking
the visible USV hull in frames captured by a center-mounted phone camera.
"""
import argparse
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def process_frames_with_mask(
    input_dir: str,
    output_dir: str,
    mask_path: str,
    camera_identifiers: list[str] | None = None,
) -> None:
    """Apply an ignore mask to frames matching configured camera identifiers.

    The black pixels in the grayscale mask become black in the output image.
    The mask is resized to the input frame size when necessary.

    Args:
        input_dir: Directory containing source JPEG or PNG frames.
        output_dir: Directory where processed frames are written.
        mask_path: Path to a grayscale ignore-mask image.
        camera_identifiers: Case-insensitive filename fragments identifying
            frames that require masking. Defaults to ``["phone", "center"]``.

    Raises:
        FileNotFoundError: If the mask image cannot be read.
        ValueError: If no JPEG or PNG images are found in ``input_dir``.
    """
    if camera_identifiers is None:
        camera_identifiers = ["phone", "center"]

    src_path = Path(input_dir)
    dst_path = Path(output_dir)
    mask_file = Path(mask_path)

    if not src_path.is_dir():
        raise NotADirectoryError(f"Input directory not found: {src_path}")

    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask image cannot be read: {mask_file}")

    image_files = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in src_path.glob(pattern)
    )
    if not image_files:
        raise ValueError(f"No JPEG or PNG images found in: {src_path}")

    dst_path.mkdir(parents=True, exist_ok=True)

    masked_count = 0
    copied_count = 0
    skipped_count = 0
    failed_writes = 0
    mask_resized = None
    mask_shape = None

    print(f"Processing {len(image_files)} image(s) from '{src_path}'.")

    for image_path in tqdm(image_files, desc="Applying ignore mask"):
        output_path = dst_path / image_path.name
        filename_lower = image_path.name.lower()
        should_apply_mask = any(
            identifier.lower() in filename_lower
            for identifier in camera_identifiers
        )

        if not should_apply_mask:
            try:
                shutil.copy2(image_path, output_path)
                copied_count += 1
            except OSError as error:
                print(f"[ERROR] Cannot copy '{image_path}': {error}")
                failed_writes += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARNING] Cannot read image, skipped: {image_path}")
            skipped_count += 1
            continue

        height, width = image.shape[:2]
        target_shape = (height, width)
        if mask_resized is None or mask_shape != target_shape:
            mask_resized = cv2.resize(mask, (width, height))
            mask_shape = target_shape

        masked_image = cv2.bitwise_and(image, image, mask=mask_resized)
        if not cv2.imwrite(str(output_path), masked_image):
            print(f"[ERROR] Cannot write masked image: {output_path}")
            failed_writes += 1
            continue

        masked_count += 1

    print("\nProcessing complete.")
    print(f"  Masked: {masked_count}")
    print(f"  Copied unchanged: {copied_count}")
    print(f"  Skipped unreadable: {skipped_count}")
    print(f"  Failed writes: {failed_writes}")
    print(f"  Output directory: {dst_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ignore-mask stage."""
    parser = argparse.ArgumentParser(
        description="Apply an ignore mask to selected dataset frames."
    )
    parser.add_argument(
        "--input-dir",
        default="data/interim/labeling_upscaled",
        help="Directory with source JPEG or PNG frames.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim/labeling_upscaled_masked_v4",
        help="Directory where processed frames are saved.",
    )
    parser.add_argument(
        "--mask-path",
        default="data/interim/phone_ignore_mask.png",
        help="Path to the grayscale ignore-mask image.",
    )
    parser.add_argument(
        "--camera-identifiers",
        nargs="+",
        default=["phone", "center"],
        help="Filename fragments identifying frames to mask.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_frames_with_mask(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mask_path=args.mask_path,
        camera_identifiers=args.camera_identifiers,
    )
