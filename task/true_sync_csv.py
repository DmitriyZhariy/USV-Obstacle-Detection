import pandas as pd
import numpy as np
import glob
import os

# ================= НАСТРОЙКИ =================
VIDEO_CSV = r"C:\Education\4 course 1 semester\Practice\Synchronization task\Data\Sync_data\manual_sync_videos.csv"            # CSV с видео (главный time)
SENSOR_FOLDER = r"C:\Education\4 course 1 semester\Practice\Synchronization task\Data\csv files"              # папка с CSV датчиков
OUTPUT_CSV = r"C:\Education\4 course 1 semester\Practice\Synchronization task\Data\Sync_data\synced_video_with_sensors.csv"

MAX_DIFF_NS = 100_000_000  # 0.1 секунды в наносекундах
# =============================================


def load_video_csv(path: str) -> pd.DataFrame:
    """
    Загружает CSV с видео.
    time приводится к int64, мусорные строки удаляются.
    """
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    df["time"] = df["time"].astype(np.int64)
    return df


def load_sensor_csv(path: str) -> pd.DataFrame:
    """
    Загружает CSV датчика, конвертирует все колонки в числовые,
    сортирует по time.
    """
    df = pd.read_csv(path, dtype=str)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = df["time"].astype(np.int64)
    return df


def nearest_with_threshold(
    video_times: np.ndarray,
    sensor_times: np.ndarray,
    sensor_values: np.ndarray
) -> np.ndarray:
    """
    Быстрый поиск ближайшего значения с порогом MAX_DIFF_NS
    """
    idx = np.searchsorted(sensor_times, video_times)

    idx_left = np.clip(idx - 1, 0, len(sensor_times) - 1)
    idx_right = np.clip(idx, 0, len(sensor_times) - 1)

    diff_left = np.abs(video_times - sensor_times[idx_left])
    diff_right = np.abs(video_times - sensor_times[idx_right])

    use_right = diff_right < diff_left
    best_idx = np.where(use_right, idx_right, idx_left)
    best_diff = np.minimum(diff_left, diff_right)

    result = sensor_values[best_idx].astype(float)
    result[best_diff > MAX_DIFF_NS] = np.nan

    return result


def main():
    print("Загрузка CSV с видео...")
    video_df = load_video_csv(VIDEO_CSV)
    video_times = video_df["time"].to_numpy()

    sensor_files = sorted(glob.glob(os.path.join(SENSOR_FOLDER, "*.csv")))
    if not sensor_files:
        raise RuntimeError("В папке sensor_data не найдено CSV файлов")

    for sensor_path in sensor_files:
        prefix = os.path.splitext(os.path.basename(sensor_path))[0]
        print(f"Синхронизация датчика: {prefix}")

        sensor_df = load_sensor_csv(sensor_path)
        sensor_times = sensor_df["time"].to_numpy()

        for col in sensor_df.columns:
            if col == "time":
                continue

            video_df[f"{prefix}_{col}"] = nearest_with_threshold(
                video_times,
                sensor_times,
                sensor_df[col].to_numpy()
            )

    video_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Готово! Итоговый CSV сохранён:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
