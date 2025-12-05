import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

class VisualVerifier:
    def __init__(self, manifest_path: str, base_height: int = 500):
        self.manifest_path = Path(manifest_path)
        self.base_height = base_height
        self.df = self._load_manifest()

    def _load_manifest(self) -> pd.DataFrame:
        df = pd.read_csv(self.manifest_path)
        time_cols = ['calc_start_time', 'calc_end_time']
        for col in time_cols:
            df[col] = pd.to_datetime(df[col])
        return df

    def get_capture_at_time(self, video_path: str, target_time: pd.Timestamp, video_start_time: pd.Timestamp):
        """
        Opens video and seeks to the specific timestamp.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        # Offset in milliseconds from the beginning of THIS video file
        offset_ms = (target_time - video_start_time).total_seconds() * 1000.0

        if offset_ms < 0:
            # We are asking for a time before this video started
            return None

        cap.set(cv2.CAP_PROP_POS_MSEC, offset_ms)
        return cap

    def find_video_covering_time(self, camera_id: str, timestamp: pd.Timestamp) -> pd.Series:
        # Find video that contains this timestamp
        subset = self.df[self.df['camera_id'] == camera_id]
        # Check overlaps
        match = subset[
            (subset['calc_start_time'] <= timestamp + timedelta(seconds=1)) &
            (subset['calc_end_time'] > timestamp)
        ]
        if not match.empty:
            return match.iloc[0]
        return None

    def resize_frame(self, frame, rotate=False):
        if frame is None:
            return np.zeros((self.base_height, int(self.base_height * 1.77), 3), dtype=np.uint8)

        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) # Or COUNTERCLOCKWISE

        h, w = frame.shape[:2]
        scale = self.base_height / h
        new_w = int(w * scale)
        return cv2.resize(frame, (new_w, self.base_height))

    def run_check(self, limit_seconds: int = 20):
        print("Starting SMART Visual Verification...")
        print("Seeking to the LATEST start time to avoid black screens.")

        left_files = self.df[self.df['camera_id'] == 'cam_left'].sort_values('calc_start_time')

        for _, left_row in left_files.iterrows():
            # 1. Determine the "Global Start Time" for this segment
            # We want to start checking when ALL valid cameras are ready

            # TEMP
            leftname = left_row['filename']
            if leftname != 'MOVI0025.avi':
                continue

            # Find candidate matches
            t_left = left_row['calc_start_time']
            phone_row = self.find_video_covering_time('phone_center', t_left)
            right_row = self.find_video_covering_time('cam_right', t_left)

            start_times = [t_left]
            if phone_row is not None: start_times.append(phone_row['calc_start_time'])
            if right_row is not None: start_times.append(right_row['calc_start_time'])

            # THE MAGIC: We start playing from the MAX of start times
            # This skips the black screen period
            play_start_time = max(start_times)

            # But we make sure we are still within the Left video file
            if play_start_time > left_row['calc_end_time']:
                print("Skipping file (overlap issue)")
                continue

            print(f"\nChecking alignment around: {play_start_time.time()}")

            # 2. Open Streams seeking to `play_start_time`
            cap_l = self.get_capture_at_time(left_row['path'], play_start_time, left_row['calc_start_time'])

            cap_p = None
            if phone_row is not None:
                cap_p = self.get_capture_at_time(phone_row['path'], play_start_time, phone_row['calc_start_time'])

            cap_r = None
            if right_row is not None:
                cap_r = self.get_capture_at_time(right_row['path'], play_start_time, right_row['calc_start_time'])

            # 3. Play
            fps = 30
            max_frames = limit_seconds * fps
            frame_cnt = 0

            while frame_cnt < max_frames:
                ret_l, frame_l = cap_l.read() if cap_l else (False, None)
                ret_p, frame_p = cap_p.read() if cap_p else (False, None)
                ret_r, frame_r = cap_r.read() if cap_r else (False, None)

                if not ret_l and not ret_p: break

                img_l = self.resize_frame(frame_l)
                img_p = self.resize_frame(frame_p, rotate=True) # Adjust rotation if needed
                img_r = self.resize_frame(frame_r)

                # Time label
                cv2.putText(img_l, f"LEFT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img_p, f"PHONE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img_r, f"RIGHT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                combined = np.hstack((img_l, img_p, img_r))
                cv2.imshow("Sync Verification (Auto-Seek)", combined)

                key = cv2.waitKey(25) & 0xFF
                if key == ord('q'): break
                elif key == 27:
                    cv2.destroyAllWindows()
                    return

                frame_cnt += 1

            if cap_l: cap_l.release()
            if cap_p: cap_p.release()
            if cap_r: cap_r.release()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    verifier = VisualVerifier("reports/synced_manifest.csv")
    verifier.run_check(limit_seconds=60)
