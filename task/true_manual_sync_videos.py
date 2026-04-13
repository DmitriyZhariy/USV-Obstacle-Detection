import os
import csv
import cv2
import math
from datetime import datetime, timezone

FPS = 30
FRAME_NS = 1_000_000_000 // FPS  # 1 кадр = 1/30 сек в наносекундах

# ------------------ ВРЕМЯ ------------------
def parse_phone_start_time_ns(filename: str) -> int:
    """Парсит начало видео телефона из имени YYYYMMDD_HHMMSS"""
    base = os.path.splitext(filename)[0]
    dt = datetime.strptime(base, "%Y%m%d_%H%M%S")
    return int(dt.timestamp() * 1_000_000_000)

def frame_to_hhmmss_ms(frame_idx: int, fps: int = FPS) -> str:
    """Преобразует кадр в hh:mm:ss.mmm"""
    total_seconds = frame_idx / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

# ------------------ VIDEO ------------------
def get_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames

def build_camera_timeline(video_files):
    """Создаёт таймлайн с локальным временем и кадрами каждого видео"""
    timeline = []
    for path in video_files:
        frames = get_frame_count(path)
        for frame_in_file in range(frames):
            timeline.append({
                "path": path,
                "frame_in_file": frame_in_file,
                "time_sec": frame_in_file / FPS
            })
    return timeline

def get_video_files(folder, ext):
    if not os.path.exists(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(ext)
    )

# ------------------ СИНХРОНИЗАЦИЯ ------------------
def sync_session(phone_video, left_videos, right_videos, session_name, csv_writer, session_offsets):
    print(f"\nСессия: {session_name}")
    print(f"  Phone: {phone_video}")
    print(f"  Left : {len(left_videos)} файлов")
    print(f"  Right: {len(right_videos)} файлов")

    offsets = session_offsets[session_name]
    first_cam = min(offsets, key=lambda k: offsets[k])
    base_start_ns = parse_phone_start_time_ns(os.path.basename(phone_video)) - int(offsets[first_cam] * 1e9)

    phone_start_ns = base_start_ns + int(offsets["phone"] * 1e9)
    left_start_ns  = base_start_ns + int(offsets["left"]  * 1e9)
    right_start_ns = base_start_ns + int(offsets["right"] * 1e9)

    phone_frames = get_frame_count(phone_video)
    left_timeline = build_camera_timeline(left_videos)
    right_timeline = build_camera_timeline(right_videos)

    phone_end_ns = phone_start_ns + phone_frames * FRAME_NS
    left_end_ns  = left_start_ns + len(left_timeline) * FRAME_NS
    right_end_ns = right_start_ns + len(right_timeline) * FRAME_NS

    global_start_ns = min(phone_start_ns, left_start_ns, right_start_ns)
    global_end_ns   = max(phone_end_ns, left_end_ns, right_end_ns)
    total_frames = int((global_end_ns - global_start_ns) / FRAME_NS)

    # ---------- CSV ----------
    for i in range(total_frames):
        t_ns = global_start_ns + i * FRAME_NS

        # PHONE
        phone_idx = (t_ns - phone_start_ns) // FRAME_NS
        if 0 <= phone_idx < phone_frames:
            phone_time = frame_to_hhmmss_ms(phone_idx)
        else:
            phone_idx = phone_time = math.nan

        # LEFT
        left_idx = (t_ns - left_start_ns) // FRAME_NS
        if 0 <= left_idx < len(left_timeline):
            left_time = frame_to_hhmmss_ms(left_timeline[left_idx]["frame_in_file"])
            left_frame = left_timeline[left_idx]["frame_in_file"]
            left_path = left_timeline[left_idx]["path"]
        else:
            left_idx = left_time = left_frame = left_path = math.nan

        # RIGHT
        right_idx = (t_ns - right_start_ns) // FRAME_NS
        if 0 <= right_idx < len(right_timeline):
            right_time = frame_to_hhmmss_ms(right_timeline[right_idx]["frame_in_file"])
            right_frame = right_timeline[right_idx]["frame_in_file"]
            right_path = right_timeline[right_idx]["path"]
        else:
            right_idx = right_time = right_frame = right_path = math.nan

        csv_writer.writerow([
            int(t_ns),
            phone_video, phone_time, phone_idx,
            left_path, left_time, left_frame,
            right_path, right_time, right_frame
        ])

# ------------------ MAIN ------------------
def main(data_root: str, output_csv: str):
    session_offsets = {
        "Session 1": {"phone": 7.0,  "left": 0.5, "right": 0.0},
        "Session 2": {"phone": 0.0,  "left": 11.0,"right": 11.5},
        "Session 3": {"phone": 57.5, "left": 0.0, "right": 0.3},
    }

    phone_root = os.path.join(data_root, "videos", "Phone")
    left_root  = os.path.join(data_root, "videos", "Left Camera")
    right_root = os.path.join(data_root, "videos", "Right Camera")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "path_phone","time_phone","frame_phone",
            "path_left","time_left","frame_left",
            "path_right","time_right","frame_right"
        ])

        for session_name in sorted(os.listdir(phone_root)):
            phone_session = os.path.join(phone_root, session_name)
            left_session  = os.path.join(left_root, session_name)
            right_session = os.path.join(right_root, session_name)

            if not os.path.isdir(phone_session):
                continue

            phone_files = get_video_files(phone_session, ".mp4")
            left_files  = get_video_files(left_session, ".avi")
            right_files = get_video_files(right_session, ".avi")

            if not phone_files or not left_files or not right_files:
                continue

            phone_video = phone_files[0]
            sync_session(phone_video, left_files, right_files, session_name, writer, session_offsets)

    print(f"\nГотово. CSV файл создан: {output_csv}")

# ------------------ ENTRY POINT ------------------
if __name__ == "__main__":
    main(data_root=r"C:\Education\4 course 1 semester\Practice\Synchronization task\Data", output_csv=r"C:\Education\4 course 1 semester\Practice\Synchronization task\Data\Sync_data\manual_sync_videos.csv")
