"""
Smart Verification tool.
Finds MULTIPLE peaks and calculates the consensus offset.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from pathlib import Path
from usv.preprocessing.audio_sync import AudioAligner

def apply_custom_filter(data, sr, lowcut, highcut):
    if data is None: return None
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0: high = 0.99
    b, a = scipy.signal.butter(5, [low, high], btype='band')
    return scipy.signal.lfilter(b, a, data)

def find_significant_peaks(sig, sr, height_threshold=0.3, distance_sec=1.0):
    """
    Finds peaks that are at least 'height_threshold' * max_amplitude.
    'distance_sec' prevents finding multiple peaks in the same clap echo.
    """
    # Normalize to 0..1
    sig_abs = np.abs(sig)
    sig_norm = sig_abs / (np.max(sig_abs) + 1e-9)

    # Find peaks
    peaks, _ = scipy.signal.find_peaks(
        sig_norm,
        height=height_threshold,
        distance=int(distance_sec * sr)
    )

    timestamps = peaks / sr
    return timestamps, sig_norm

def verify_smart_peaks(ref_path, tgt_path):
    print(f"Smart Analysis for:\nRef: {ref_path}\nTgt: {tgt_path}\n")

    aligner = AudioAligner(sample_rate=16000)
    duration = 35 # Чуть больше, чтобы захватить оба хлопка наверняка

    ref_sig = aligner.load_audio(Path(ref_path), duration=duration)
    tgt_sig = aligner.load_audio(Path(tgt_path), duration=duration)

    # Фильтр (оставляем только щелчки)
    ref_clean = apply_custom_filter(ref_sig, aligner.sr, 3000, 7900)
    tgt_clean = apply_custom_filter(tgt_sig, aligner.sr, 3000, 7900)

    # Ищем ВСЕ значимые пики (порог 0.2 от макс громкости)
    t_ref, y_ref = find_significant_peaks(ref_clean, aligner.sr, height_threshold=0.2)
    t_tgt, y_tgt = find_significant_peaks(tgt_clean, aligner.sr, height_threshold=0.2)

    print(f"Phone Peaks found at: {t_ref}")
    print(f"Cam Peaks found at:   {t_tgt}")
    print("-" * 30)

    # Матрица разностей: проверяем гипотезу для каждой пары
    # Мы ищем такую разницу dT, которая объясняет несколько пар пиков
    possible_offsets = []

    print("Testing pairs:")
    for t_r in t_ref:
        for t_c in t_tgt:
            diff = t_r - t_c
            print(f"  Ref({t_r:.2f}) - Cam({t_c:.2f}) = {diff:.4f}s")
            possible_offsets.append(diff)

    # Ищем кластеры (значения, которые близки друг к другу)
    # Если разница -7.96 реальна, она встретится столько раз, сколько было хлопков
    possible_offsets = np.array(possible_offsets)

    # Грубая кластеризация: округляем до 0.1с и ищем моду
    rounded = np.round(possible_offsets, 1)
    unique, counts = np.unique(rounded, return_counts=True)
    best_idx = np.argmax(counts)
    consensus_offset = unique[best_idx]

    print("-" * 30)
    print(f"Most frequent offset (Consensus): ~{consensus_offset} sec")

    # Более точный расчет: берем среднее из кластера победителя
    cluster = possible_offsets[np.abs(possible_offsets - consensus_offset) < 0.2]
    precise_offset = np.mean(cluster)
    print(f"Precise Average Offset: {precise_offset:.4f} sec")
    print("-" * 30)

    # Визуализация
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.title("Phone Peaks")
    plt.plot(np.linspace(0, duration, len(y_ref)), y_ref)
    plt.plot(t_ref, y_ref[(t_ref * aligner.sr).astype(int)], 'rx', markersize=12)

    plt.subplot(2, 1, 2)
    plt.title("Camera Peaks")
    plt.plot(np.linspace(0, duration, len(y_tgt)), y_tgt, color='orange')
    plt.plot(t_tgt, y_tgt[(t_tgt * aligner.sr).astype(int)], 'rx', markersize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ref = "data/raw/center_phone/VID_20251027_131854.mp4"
    tgt = "data/raw/left_cam/MOVI0000.avi"
    verify_smart_peaks(ref, tgt)
