"""
Debugging script to visualize audio synchronization results.
Plots the reference (Phone) and target (Dashcam) audio waveforms aligned by the calculated offset.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from usv.preprocessing.audio_sync import AudioAligner

def visualize_alignment(manifest_path: str, raw_data_dir: str):
    df = pd.read_csv(manifest_path)
    base_dir = Path(raw_data_dir)
    aligner = AudioAligner(sample_rate=22050)

    # 1. Find the Master Phone video
    phone_row = df[df['camera_id'] == 'phone_center'].sort_values('filename').iloc[0]

    # 2. Iterate over dashcams
    for cam_id in ['cam_left', 'cam_right']:
        cam_rows = df[df['camera_id'] == cam_id].sort_values('filename')
        if cam_rows.empty:
            continue

        target_row = cam_rows.iloc[0]

        print(f"--- Visualizing {cam_id} ---")
        print(f"Reference: {phone_row['filename']}")
        print(f"Target:    {target_row['filename']}")

        # Load audio (first 60 seconds is usually enough for the clap)
        # Note: Increase duration if the clap happened later
        duration = 60
        ref_audio = aligner.load_audio(Path(phone_row['path']), duration=duration)
        tgt_audio = aligner.load_audio(Path(target_row['path']), duration=duration)

        # Recalculate offset (or read from CSV if we stored it, but recalc is safer for debug)
        offset, corr = aligner.find_offset(ref_audio, tgt_audio)
        print(f"Calculated Offset: {offset:.4f}s")

        # --- PLOTTING ---
        plt.figure(figsize=(12, 6))

        # Create time axes
        time_ref = np.linspace(0, len(ref_audio)/aligner.sr, len(ref_audio))
        time_tgt = np.linspace(0, len(tgt_audio)/aligner.sr, len(tgt_audio))

        # Shift target time axis by the offset
        # If offset is positive (Target is later), we add offset to its time
        # to align it visually with Reference.
        time_tgt_shifted = time_tgt + offset

        plt.subplot(2, 1, 1)
        plt.title(f"Alignment Check: {cam_id} (Offset: {offset:.2f}s)")
        plt.plot(time_ref, ref_audio, label='Phone (Master)', color='blue', alpha=0.7)
        plt.plot(time_tgt_shifted, tgt_audio, label=f'{cam_id} (Shifted)', color='orange', alpha=0.7)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.ylabel("Amplitude")

        # Zoom in on the "Clap" area (optional, finding the max peak)
        # Let's just show the first 10 seconds after the offset to see the start
        plt.subplot(2, 1, 2)
        plt.title("Zoomed View (Start of overlap)")
        plt.plot(time_ref, ref_audio, label='Phone', color='blue', alpha=0.7)
        plt.plot(time_tgt_shifted, tgt_audio, label=f'{cam_id}', color='orange', alpha=0.7)

        # Set x-limit to zoom around the start of the second video
        start_zoom = max(0, offset - 5)
        end_zoom = start_zoom + 10
        plt.xlim(start_zoom, end_zoom)
        plt.grid(True)
        plt.xlabel("Time (seconds)")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_alignment("reports/synced_manifest.csv", "data/raw")
