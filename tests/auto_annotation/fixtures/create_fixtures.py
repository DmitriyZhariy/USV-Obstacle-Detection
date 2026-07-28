"""Run once to generate synthetic fixture data for tests."""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent
CLIP_NAME = "test_left_FIXTURE_0001"
N_FRAMES = 3
H, W = 64, 64

frames_dir = FIXTURE_DIR / "frames" / CLIP_NAME
frames_dir.mkdir(parents=True, exist_ok=True)

for i in range(1, N_FRAMES + 1):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:, :, i - 1] = 128  # R/G/B distinguishable per frame
    cv2.imwrite(str(frames_dir / f"{i:04d}.jpeg"), frame)

meta = pd.DataFrame({
    "local_seq_idx": [1, 2, 3],
    "original_video": ["FIXTURE.avi"] * 3,
    "original_frame_idx": [100, 106, 112],
})
meta_dir = FIXTURE_DIR / "metadata"
meta_dir.mkdir(exist_ok=True)
meta.to_csv(meta_dir / f"{CLIP_NAME}.csv", index=False)
print("Fixtures created.")