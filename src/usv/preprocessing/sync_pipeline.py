"""
Pipeline for synchronizing dashcam videos with the master phone timeline
using audio cross-correlation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import List, Optional

# Import our custom tools
from usv.preprocessing.audio_sync import AudioAligner

class SyncPipeline:
    def __init__(self, manifest_path: str, raw_data_dir: str):
        self.manifest_path = Path(manifest_path)
        self.raw_data_dir = Path(raw_data_dir)
        self.aligner = AudioAligner(sample_rate=22050)

    def load_manifest(self) -> pd.DataFrame:
        df = pd.read_csv(self.manifest_path)
        # Convert string timestamps back to datetime objects
        cols = ['filename_timestamp', 'calc_start_time', 'calc_end_time']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    def find_anchor_offset(self,
                           ref_row: pd.Series,
                           tgt_row: pd.Series,
                           search_window_sec: int = 120) -> float:
        """
        Finds temporal offset between a reference video (Phone) and target (Dashcam).
        Only checks the first 'search_window_sec' seconds to save memory.
        """
        print(f"Syncing {tgt_row['filename']} against {ref_row['filename']}...")

        ref_path = Path(ref_row['path'])
        tgt_path = Path(tgt_row['path'])

        # Load audio (only first N seconds)
        # We assume they started roughly at the same time (+- 1-2 mins)
        ref_audio = self.aligner.load_audio(ref_path, duration=search_window_sec)
        tgt_audio = self.aligner.load_audio(tgt_path, duration=search_window_sec)

        if ref_audio is None or tgt_audio is None:
            print("Error loading audio.")
            return 0.0

        # Calculate offset
        offset, correlation_val = self.aligner.find_offset(ref_audio, tgt_audio, filter_audio=True)

        print(f"  Found Offset: {offset:.4f}s (Corr: {correlation_val:.2f})")
        return offset

    def run(self, output_path: str = "reports/synced_manifest.csv"):
        df = self.load_manifest()

        # 1. Identify Master Start Time (Phone)
        # We take the very first phone video as the absolute anchor
        phone_videos = df[df['camera_id'] == 'phone_center'].sort_values('calc_start_time')
        if phone_videos.empty:
            raise ValueError("No phone videos found in manifest!")

        master_ref = phone_videos.iloc[0]
        master_start_time = master_ref['calc_start_time']
        print(f"Master Anchor: {master_ref['filename']} starts at {master_start_time}")

        # 2. Process each dashcam (Left and Right) separately
        for cam_id in ['cam_left', 'cam_right']:
            cam_videos = df[df['camera_id'] == cam_id].sort_values('filename')

            if cam_videos.empty:
                print(f"No videos found for {cam_id}, skipping.")
                continue

            # We take the FIRST video of the dashcam stream to sync
            first_cam_vid = cam_videos.iloc[0]

            # Find offset relative to the master phone video
            # Offset > 0: Cam starts AFTER Phone
            # Offset < 0: Cam starts BEFORE Phone
            offset = self.find_anchor_offset(master_ref, first_cam_vid)

            # Calculate absolute start time for the first dashcam video
            # T_cam = T_phone + offset
            first_cam_start_time = master_start_time + timedelta(seconds=offset)

            print(f"  => {cam_id} sequence starts at {first_cam_start_time}")

            # 3. Propagate timestamps to all videos in this stream
            # We assume files are continuous (End of file N = Start of file N+1)
            # NOTE: Dashcams usually have a small overlap or gap (1-2s).
            # For now, we assume 0 gap. Later we can refine this.

            current_start_time = first_cam_start_time

            for idx, row in cam_videos.iterrows():
                duration = row['duration_sec']

                # Update DataFrame
                df.at[idx, 'calc_start_time'] = current_start_time
                df.at[idx, 'calc_end_time'] = current_start_time + timedelta(seconds=duration)

                # Update start time for the NEXT video
                # (Assuming continuous recording)
                current_start_time += timedelta(seconds=duration)

        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_path, index=False)
        print(f"Synced manifest saved to {output_path}")

if __name__ == "__main__":
    # Adjust paths as needed
    manifest_file = "reports/data_manifest.csv"
    data_dir = "data/raw"

    pipeline = SyncPipeline(manifest_file, data_dir)
    pipeline.run()
