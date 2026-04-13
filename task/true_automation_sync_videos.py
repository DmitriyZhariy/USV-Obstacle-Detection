import os
import glob
import csv
import numpy as np
import librosa
from scipy.signal import correlate
from moviepy import VideoFileClip

# ================= НАСТРОЙКИ =================
DATA_ROOT = r"C:\Education\4 course 1 semester\Practice\Synchronization task\Data\videos"     # корень датасета
OUTPUT_CSV = r"C:\Education\4 course 1 semester\Practice\Synchronization task\Data\Sync_data\automation_sync_videos.csv"

SAMPLE_RATE = 16000
SYNC_WINDOW = 30.0            # секунд для корреляции

# примерные времена хлопка (секунды)
PHONE_CLAP_TIME = 14.0
LEFT_CLAP_TIME = 22.0
RIGHT_CLAP_TIME = 21.0
# =============================================


def extract_audio_window(video_path, center_time, window=30.0, sr=16000):
    """
    Извлекает аудио окно вокруг center_time
    БЕЗ использования subclip (важно!)
    """
    clip = VideoFileClip(video_path)

    audio = clip.audio.to_soundarray(fps=sr)
    clip.close()

    audio = audio.mean(axis=1)  # mono

    center_sample = int(center_time * sr)
    half = int((window / 2) * sr)

    start = max(0, center_sample - half)
    end = min(len(audio), center_sample + half)

    return audio[start:end]


def find_offset(ref_audio, target_audio, sr):
    """
    Возвращает сдвиг target относительно ref (в секундах)
    """
    corr = correlate(target_audio, ref_audio, mode="full")
    lag = np.argmax(corr) - len(ref_audio) + 1
    return lag / sr


def process_session(session_id, writer):
    print(f"\n=== Session {session_id} ===")

    phone_videos = sorted(glob.glob(
        os.path.join(DATA_ROOT, "Phone", f"Session {session_id+1}", "*.mp4")
    ))
    phone_path = phone_videos[session_id]

    # опорное аудио из Phone
    ref_audio = extract_audio_window(
        phone_path,
        center_time=PHONE_CLAP_TIME,
        window=SYNC_WINDOW
    )

    left_videos = sorted(glob.glob(
        os.path.join(DATA_ROOT, "Left camera", f"Session {session_id+1}", "*.avi")
    ))
    right_videos = sorted(glob.glob(
        os.path.join(DATA_ROOT, "Right camera", f"Session {session_id+1}", "*.avi")
    ))

    for left_path, right_path in zip(left_videos, right_videos):

        left_audio = extract_audio_window(
            left_path,
            center_time=LEFT_CLAP_TIME,
            window=SYNC_WINDOW
        )
        right_audio = extract_audio_window(
            right_path,
            center_time=RIGHT_CLAP_TIME,
            window=SYNC_WINDOW
        )

        left_offset = find_offset(ref_audio, left_audio, SAMPLE_RATE)
        right_offset = find_offset(ref_audio, right_audio, SAMPLE_RATE)

        time_phone = PHONE_CLAP_TIME
        time_left = LEFT_CLAP_TIME + left_offset
        time_right = RIGHT_CLAP_TIME + right_offset

        print(
            f"{os.path.basename(left_path)} | "
            f"L: {time_left:.3f}s | R: {time_right:.3f}s"
        )

        writer.writerow([
            phone_path,
            round(time_phone, 3),
            left_path,
            round(time_left, 3),
            right_path,
            round(time_right, 3)
        ])


# =================== MAIN ====================

if __name__ == "__main__":

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path_phone",
            "time_video_phone",
            "path_left",
            "time_video_left",
            "path_right",
            "time_video_right"
        ])

        for session_id in range(3):
            process_session(session_id, writer)

    print("\n✅ Готово! CSV сохранён в:", OUTPUT_CSV)
