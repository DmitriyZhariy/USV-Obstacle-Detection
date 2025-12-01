"""
Module to parse video files and generate a temporal manifest.
Extracts duration, resolution, and timestamps from filenames/metadata.
"""
import re
import pandas as pd
import ffmpeg
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

class VideoManifestGenerator:
    def __init__(self, raw_data_dir: str):
        self.root_dir = Path(raw_data_dir)
        # Regex for phone filenames: VID_YYYYMMDD_HHMMSS.mp4
        # Capture groups: (Year)(Month)(Day)_(Hour)(Min)(Sec)
        self.phone_pattern = re.compile(r'VID_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})')

    def get_video_duration(self, file_path: Path) -> float:
        """Uses ffprobe to get exact duration in seconds."""
        try:
            probe = ffmpeg.probe(str(file_path))
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream:
                return float(video_stream['duration'])
        except Exception as e:
            print(f"Error probing {file_path.name}: {e}")
        return 0.0

    def parse_phone_filename(self, filename: str) -> Optional[datetime]:
        """Converts filename timestamp to datetime object."""
        match = self.phone_pattern.search(filename)
        if match:
            y, m, d, H, M, S = map(int, match.groups())
            return datetime(y, m, d, H, M, S)
        return None

    def create_manifest(self, output_csv: str = "dataset_manifest.csv"):
        """
        Scans directory and builds a Pandas DataFrame.

        Logic for Phone Camera:
            Filename timestamp = END time of the recording.
            Start time = End time - Duration.
        """
        records = []

        # Recursively find all mp4 files
        files = sorted(list(self.root_dir.rglob("*.mp4")))
        print(f"Found {len(files)} video files.")

        for f in files:
            # Determine camera type based on folder name or filename convention
            cam_type = "unknown"
            if "phone" in str(f).lower() or "vid_" in f.name.lower():
                cam_type = "phone_center"
            elif "left" in str(f).lower():
                cam_type = "cam_left"
            elif "right" in str(f).lower():
                cam_type = "cam_right"

            duration = self.get_video_duration(f)
            file_ts = self.parse_phone_filename(f.name)

            start_time = None
            end_time = None

            if file_ts and cam_type == "phone_center":
                # Filename represents the END time
                end_time = file_ts
                start_time = end_time - timedelta(seconds=duration)
            else:
                # Logic for dashcams (if they have timestamps in names) can be added here later
                pass

            records.append({
                "filename": f.name,
                "path": str(f),
                "camera_id": cam_type,
                "duration_sec": duration,
                "filename_timestamp": file_ts,
                "calc_start_time": start_time,
                "calc_end_time": end_time
            })

        df = pd.DataFrame(records)

        # Sort by camera and time
        if not df.empty and 'filename_timestamp' in df.columns:
            df = df.sort_values(by=['camera_id', 'filename_timestamp'])

        # Create output directory if it doesn't exist
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_csv, index=False)
        print(f"Manifest saved to {output_csv}")
        return df

if __name__ == "__main__":
    # Path to your raw data
    data_path = Path("data/raw")

    if data_path.exists():
        gen = VideoManifestGenerator(str(data_path))
        df = gen.create_manifest("reports/data_manifest.csv")

        # Print first few rows to verify
        print(df[['filename', 'duration_sec', 'calc_start_time']].head())
    else:
        print(f"Path {data_path} does not exist. Please check your structure.")
