import os
import csv
import cv2
import math
from datetime import datetime

# ================== НАСТРОЙКИ ==================

FPS = 30
FRAME_NS = 1_000_000_000 // FPS  # 1 кадр в наносекундах

# ================== ВРЕМЯ ==================

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

# ================== VIDEO ==================

def get_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames

# ================== СИНХРОНИЗАЦИЯ ==================

def sync_session(phone_video, left_video, right_video, session_name, csv_writer, session_offsets):
    print(f"\nСессия: {session_name}")
    print(f"  Phone: {phone_video}")
    print(f"  Left : {left_video}")
    print(f"  Right: {right_video}")

    # ---------- Получаем offsets (в секундах) ----------
    offset_phone = session_offsets[session_name]["phone"]
    offset_left  = session_offsets[session_name]["left"]
    offset_right = session_offsets[session_name]["right"]

    print(f"  Offset Phone: {offset_phone:.3f} s")
    print(f"  Offset Left : {offset_left:.3f} s")
    print(f"  Offset Right: {offset_right:.3f} s")

    # ---------- VIDEO ----------
    phone_frames = get_frame_count(phone_video)
    left_frames  = get_frame_count(left_video)
    right_frames = get_frame_count(right_video)

    phone_start_ns = parse_phone_start_time_ns(os.path.basename(phone_video)) + int(offset_phone * 1e9)
    left_start_ns  = parse_phone_start_time_ns(os.path.basename(phone_video)) + int(offset_left  * 1e9)
    right_start_ns = parse_phone_start_time_ns(os.path.basename(phone_video)) + int(offset_right * 1e9)

    phone_end_ns = phone_start_ns + phone_frames * FRAME_NS
    left_end_ns  = left_start_ns  + left_frames  * FRAME_NS
    right_end_ns = right_start_ns + right_frames * FRAME_NS

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
        if 0 <= left_idx < left_frames:
            left_time = frame_to_hhmmss_ms(left_idx)
        else:
            left_idx = left_time = math.nan

        # RIGHT
        right_idx = (t_ns - right_start_ns) // FRAME_NS
        if 0 <= right_idx < right_frames:
            right_time = frame_to_hhmmss_ms(right_idx)
        else:
            right_idx = right_time = math.nan

        csv_writer.writerow([
            int(t_ns),
            phone_video, phone_time, phone_idx,
            left_video, left_time, left_idx,
            right_video, right_time, right_idx
        ])

# ================== MAIN ==================

def main(data_root: str, output_csv: str):
    # ---------- Ручные offsets для точной синхронизации ----------
    session_offsets = {
        "Session 1": {"Phone": 7.0,  "left": 0.5, "right": 0.0},
        "Session 2": {"Phone": 0.0,  "left": 11.0,"right": 11.5},
        "Session 3": {"Phone": 57.5, "left": 0.0, "right": 0.3},
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

            phone_files = [f for f in os.listdir(phone_session) if f.lower().endswith(".mp4")]
            left_files  = [f for f in os.listdir(left_session)  if f.lower().endswith(".avi")]
            right_files = [f for f in os.listdir(right_session) if f.lower().endswith(".avi")]

            if not phone_files or not left_files or not right_files:
                continue

            phone_video = os.path.join(phone_session, phone_files[0])
            left_video  = os.path.join(left_session, sorted(left_files)[0])
            right_video = os.path.join(right_session, sorted(right_files)[0])

            sync_session(phone_video, left_video, right_video, session_name, writer, session_offsets)

    print(f"\nГотово. CSV файл создан: {output_csv}")

# ================== ENTRY POINT ==================

if __name__ == "__main__":
    main(data_root="Data", output_csv="synced_videos.csv")
