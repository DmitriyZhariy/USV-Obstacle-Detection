"""
Pipeline for synchronizing dashcam videos with the master phone timeline.
"""
import pandas as pd
from pathlib import Path
from datetime import timedelta
from usv.preprocessing.audio_sync import AudioAligner

class SyncPipeline:
    def __init__(self, manifest_path: str, raw_data_dir: str):
        self.manifest_path = Path(manifest_path)
        self.raw_data_dir = Path(raw_data_dir)
        # 16kHz is native for dashcams
        self.aligner = AudioAligner(sample_rate=16000)

        # DERIVED CONSTANTS
        self.HARDWARE_LATENCY = 0.0 # seconds (Removed manual correction)

    def load_manifest(self) -> pd.DataFrame:
        df = pd.read_csv(self.manifest_path)
        cols = ['filename_timestamp', 'calc_start_time', 'calc_end_time']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    def run(self, output_path: str = "reports/synced_manifest.csv"):
        df = self.load_manifest()

        # 1. Master Anchor (Phone)
        phone_videos = df[df['camera_id'] == 'phone_center'].sort_values('calc_start_time')
        if phone_videos.empty: raise ValueError("No phone videos!")
        master_ref = phone_videos.iloc[0]
        master_start_time = master_ref['calc_start_time']

        print(f"Master Anchor: {master_ref['filename']}")

        # 2. Process Dashcams
        for cam_id in ['cam_left', 'cam_right']:
            cam_videos = df[df['camera_id'] == cam_id].sort_values('filename')
            if cam_videos.empty: continue

            # Sync first video
            first_cam_vid = cam_videos.iloc[0]

            # Load audio
            print(f"Syncing {cam_id}...")
            ref_audio = self.aligner.load_audio(Path(master_ref['path']), duration=40) # 40s to catch both claps
            tgt_audio = self.aligner.load_audio(Path(first_cam_vid['path']), duration=40)

            # Calculate Audio Offset
            # We use apply_cleaning=True (Bandpass + Denoise)
            audio_offset, corr = self.aligner.find_offset(ref_audio, tgt_audio, apply_cleaning=True)

            # Apply Hardware Correction
            total_offset = audio_offset + self.HARDWARE_LATENCY

            print(f"  Audio Offset: {audio_offset:.4f}s")
            print(f"  Latency Corr: {self.HARDWARE_LATENCY:.4f}s")
            print(f"  Final Offset: {total_offset:.4f}s")

            # Propagate time
            first_cam_start_time = master_start_time + timedelta(seconds=total_offset)
            current_start_time = first_cam_start_time

            for idx, row in cam_videos.iterrows():
                duration = row['duration_sec']
                df.at[idx, 'calc_start_time'] = current_start_time
                df.at[idx, 'calc_end_time'] = current_start_time + timedelta(seconds=duration)
                current_start_time += timedelta(seconds=duration)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    pipeline = SyncPipeline("reports/data_manifest.csv", "data/raw")
    pipeline.run()
