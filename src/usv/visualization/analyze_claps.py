"""
Diagnostic tool to analyze why Cross-Correlation fails to match Ground Truth.
Plots waveforms, envelopes, and onsets aligned by manual ground truth.
"""
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from pathlib import Path
from usv.preprocessing.audio_sync import AudioAligner

def analyze_signals(ref_path, tgt_path, manual_offset):
    aligner = AudioAligner(sample_rate=48000) # Максимальное качество для анализа

    print("Loading audio...")
    # Грузим чуть больше, чтобы был запас для сдвига
    duration = 15
    ref_sig = aligner.load_audio(Path(ref_path), duration=duration)
    tgt_sig = aligner.load_audio(Path(tgt_path), duration=duration)

    # Применяем фильтры (как в пайплайне)
    ref_sig = aligner.clean_signal(ref_sig)
    tgt_sig = aligner.clean_signal(tgt_sig)

    # --- СИНХРОНИЗАЦИЯ ПО GROUND TRUTH ---
    # Сдвигаем Target сигнал на manual_offset
    # Если offset отрицательный (-7.96), значит Target начался РАНЬШЕ.
    # Нам нужно "отрезать" начало у Target, чтобы совместить с Ref.

    offset_samples = int(abs(manual_offset) * aligner.sr)

    # Обрезаем для выравнивания
    if manual_offset < 0:
        tgt_aligned = tgt_sig[offset_samples:]
        ref_aligned = ref_sig[:len(tgt_aligned)]
    else:
        ref_aligned = ref_sig[offset_samples:]
        tgt_aligned = tgt_sig[:len(ref_aligned)]

    # Оставим только 1 секунду вокруг самого громкого места (хлопка)
    # Чтобы рассмотреть детали в микроскоп
    peak_idx = np.argmax(np.abs(ref_aligned))
    window = aligner.sr // 2 # 0.5 sec window
    start = max(0, peak_idx - window)
    end = min(len(ref_aligned), peak_idx + window)

    ref_zoom = ref_aligned[start:end]
    tgt_zoom = tgt_aligned[start:end]
    time_axis = np.linspace(0, len(ref_zoom)/aligner.sr, len(ref_zoom))

    # --- ВЫЧИСЛЕНИЕ ОГИБАЮЩЕЙ (ENVELOPE) ---
    # Hilbert transform дает точную амплитудную огибающую
    from scipy.signal import hilbert
    ref_env = np.abs(hilbert(ref_zoom))
    tgt_env = np.abs(hilbert(tgt_zoom))

    # --- ВЫЧИСЛЕНИЕ ONSET (Начало атаки) ---
    onset_env_ref = librosa.onset.onset_strength(y=ref_zoom, sr=aligner.sr)
    onset_env_tgt = librosa.onset.onset_strength(y=tgt_zoom, sr=aligner.sr)
    # Растягиваем onsets обратно к временной шкале (так как они посчитаны по фреймам)
    times_onset = librosa.times_like(onset_env_ref, sr=aligner.sr)

    # --- ПЛОТТИНГ ---
    plt.figure(figsize=(14, 10))

    # 1. Сырая волна (Waveform)
    plt.subplot(3, 1, 1)
    plt.title(f"Waveforms Aligned by {manual_offset}s (Zoomed)")
    plt.plot(time_axis, ref_zoom / np.max(np.abs(ref_zoom)), label="Phone (Ref)", alpha=0.7)
    plt.plot(time_axis, tgt_zoom / np.max(np.abs(tgt_zoom)), label="Cam (Tgt)", alpha=0.7)
    plt.legend()
    plt.grid(True)

    # 2. Огибающая (Envelope) - показывает энергию
    plt.subplot(3, 1, 2)
    plt.title("Amplitude Envelopes (Hilbert)")
    plt.plot(time_axis, ref_env / np.max(ref_env), label="Phone Envelope", linestyle='--')
    plt.plot(time_axis, tgt_env / np.max(tgt_env), label="Cam Envelope", linestyle='--')
    plt.legend()
    plt.grid(True)

    # 3. Спектрограмма (Разница частот)
    plt.subplot(3, 1, 3)
    plt.title("Onset Strength (Sharpness of Attack)")
    plt.plot(times_onset, onset_env_ref / np.max(onset_env_ref), label="Phone Onset")
    plt.plot(times_onset, onset_env_tgt / np.max(onset_env_tgt), label="Cam Onset")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("reports/signal_analysis.png")
    print("Analysis saved to reports/signal_analysis.png")
    plt.show()

if __name__ == "__main__":
    # Укажи пути
    ref = "data/raw/center_phone/VID_20251027_131854.mp4"
    tgt = "data/raw/left_cam/MOVI0000.avi"
    manual_offset = -7.96 # Твой идеал

    analyze_signals(ref, tgt, manual_offset)
