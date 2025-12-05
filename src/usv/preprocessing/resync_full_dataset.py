"""
Advanced Pipeline: Synchronizes EVERY individual video file against the FULL Phone Timeline.
Automatically switches between phone files as the master reference.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import librosa
import warnings

from usv.preprocessing.audio_sync import AudioAligner

class FullDatasetResync:
    def __init__(self, manifest_path: str, raw_data_dir: str):
        self.manifest_path = Path(manifest_path)
        self.aligner = AudioAligner(sample_rate=16000)
        self.SEARCH_WINDOW_SEC = 25 # Расширил окно поиска

    def load_manifest(self) -> pd.DataFrame:
        df = pd.read_csv(self.manifest_path)
        cols = ['filename_timestamp', 'calc_start_time', 'calc_end_time']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    def get_master_audio_segment(self,
                                 phone_df: pd.DataFrame,
                                 target_global_time: pd.Timestamp,
                                 duration_needed: float):
        """
        Smart Lookup: Finds which phone file covers the requested timestamp
        and extracts the specific audio chunk.
        """
        # Ищем файл телефона, который содержит нужный нам момент времени (плюс-минус буфер)
        # Нам нужно, чтобы файл телефона покрывал хотя бы начало нашего окна поиска

        # Фильтруем файлы, которые заканчиваются ПОСЛЕ начала нашего окна
        candidates = phone_df[phone_df['calc_end_time'] > target_global_time]

        if candidates.empty:
            return None, None # Мы вышли за пределы всех записей телефона

        # Берем первый подходящий (самый ранний из тех, что заканчиваются позже)
        # Обычно это тот самый файл
        ref_row = candidates.sort_values('calc_start_time').iloc[0]

        # Проверяем, не начинается ли он слишком поздно (дырка в записи телефона)
        if ref_row['calc_start_time'] > target_global_time + timedelta(seconds=20):
             # Если следующий файл телефона начинается через 20 сек после того места, где мы ищем
             # значит мы попали в дырку (Gap) между записями телефона.
             return None, None

        # Вычисляем смещение внутри этого файла
        # offset = (Время_Поиска - Начало_Файла_Телефона)
        file_start = ref_row['calc_start_time']
        offset_in_file = (target_global_time - file_start).total_seconds()

        # Если offset отрицательный (мы ищем чуть раньше начала файла), librosa не поймет.
        # Ставим 0.
        load_offset = max(0.0, offset_in_file)

        # Загружаем
        path = str(ref_row['path'])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                y, _ = librosa.load(
                    path,
                    sr=16000,
                    offset=load_offset,
                    duration=duration_needed,
                    mono=True
                )
                return y, ref_row['filename']
            except Exception as e:
                print(f"  [Error] Failed to load phone audio from {path}: {e}")
                return None, None

    def run(self, output_path: str = "reports/synced_manifest_full.csv"):
        df = self.load_manifest()

        # 1. Prepare Master Timeline (All Phone Videos)
        phone_videos = df[df['camera_id'] == 'phone_center'].sort_values('calc_start_time')
        if phone_videos.empty: raise ValueError("No phone videos!")

        # Глобальный ноль - начало самой первой записи телефона
        global_zero = phone_videos.iloc[0]['calc_start_time']
        print(f"Global Timeline Start: {global_zero}")

        # 2. Process Each Camera Stream
        for cam_id in ['cam_left', 'cam_right']:
            print(f"\n=== Resyncing Stream: {cam_id} ===")
            cam_videos = df[df['camera_id'] == cam_id].sort_values('filename')

            # Мы храним "ожидаемое время старта" как абсолютный Timestamp
            # Для первого файла мы не знаем, где он, поэтому это будет None, и мы будем искать с начала
            expected_start_time = global_zero

            is_first_file = True

            for idx, row in cam_videos.iterrows():
                tgt_path = Path(row['path'])
                duration = row['duration_sec']

                # --- STEP A: Prepare Dashcam Audio ---
                sample_dur = 40.0
                if duration < sample_dur: sample_dur = duration

                tgt_audio = self.aligner.load_audio(tgt_path, duration=sample_dur)

                can_sync = False
                found_offset = 0.0

                if tgt_audio is not None and len(tgt_audio) > 0:
                    # --- STEP B: Define Search Window ---
                    # Где мы ищем этот файл на таймлайне телефона?

                    if is_first_file:
                        # Первый файл ищем в самом начале 1-го видео телефона
                        search_center = global_zero
                        search_start_time = global_zero # Ищем с 0
                        search_duration = 120.0 # Широкое окно для старта
                    else:
                        # Остальные ищем вокруг ожидаемого времени (конец предыдущего)
                        search_center = expected_start_time
                        # Начало окна поиска = Ожидание - 20 сек
                        search_start_time = search_center - timedelta(seconds=self.SEARCH_WINDOW_SEC)
                        # Длительность окна = 20 (до) + 40 (сэмпл) + 20 (после)
                        search_duration = self.SEARCH_WINDOW_SEC + sample_dur + self.SEARCH_WINDOW_SEC

                    # --- STEP C: Get Master Audio (Smart Lookup) ---
                    # Функция сама найдет нужный файл телефона
                    ref_audio, ref_filename = self.get_master_audio_segment(
                        phone_videos,
                        search_start_time,
                        search_duration
                    )

                    if ref_audio is not None and len(ref_audio) > 0:
                        # --- STEP D: Find Offset ---
                        # local_offset - это сдвиг относительно НАЧАЛА ЗАГРУЖЕННОГО КУСКА (y_ref)
                        local_offset, corr = self.aligner.find_offset(ref_audio, tgt_audio, apply_cleaning=True)

                        # Если корреляция слишком слабая, не верим ей
                        if corr > 10.0: # Порог зависит от метода, для обычной корреляции цифры большие
                            # Абсолютное время старта = Время начала окна поиска + сдвиг
                            real_start_time = search_start_time + timedelta(seconds=local_offset)

                            print(f"File {row['filename']}: Synced with {ref_filename} (Corr: {corr:.0f})")
                            # print(f"  -> Delta from expected: {(real_start_time - expected_start_time).total_seconds():.2f}s")

                            can_sync = True
                        else:
                            print(f"File {row['filename']}: Low correlation ({corr:.1f}). Fallback.")
                    else:
                        print(f"File {row['filename']}: No overlapping phone audio found (Gap?). Fallback.")

                # --- STEP E: Update DataFrame ---

                if can_sync:
                    # Используем найденное время
                    final_start = real_start_time
                else:
                    # Fallback: просто стыкуем к предыдущему
                    final_start = expected_start_time
                    if not is_first_file:
                        print(f"  -> Estimation (Docking to previous)")

                df.at[idx, 'calc_start_time'] = final_start
                df.at[idx, 'calc_end_time'] = final_start + timedelta(seconds=duration)

                # Обновляем ожидание для следующего (Конец текущего)
                expected_start_time = final_start + timedelta(seconds=duration)
                is_first_file = False

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_path, index=False)
        print(f"Saved FULL SYNC manifest to {output_path}")

if __name__ == "__main__":
    resyncer = FullDatasetResync("reports/data_manifest.csv", "data/raw")
    resyncer.run()
