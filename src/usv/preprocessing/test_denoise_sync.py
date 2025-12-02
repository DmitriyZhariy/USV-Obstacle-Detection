import librosa
import numpy as np
import scipy.signal
import noisereduce as nr
from pathlib import Path
import matplotlib.pyplot as plt

# Импортируем наш класс, чтобы не дублировать код загрузки
from usv.preprocessing.audio_sync import AudioAligner

def test_denoising(ref_path, tgt_path):
    print(f"Loading files...\nRef: {ref_path}\nTgt: {tgt_path}")

    # Используем 16000 Hz (родная частота)
    aligner = AudioAligner(sample_rate=16000)

    # Грузим первые 20 секунд (достаточно для хлопка)
    ref_audio = aligner.load_audio(Path(ref_path), duration=20)
    tgt_audio = aligner.load_audio(Path(tgt_path), duration=20)

    if ref_audio is None or tgt_audio is None:
        print("Error loading audio")
        return

    print("Applying Spectral Gating Denoising...")
    # Считаем, что шум постоянен (мотор/ветер)
    # Берем профиль шума из первых 0.5 секунд видео (до хлопка)
    # prop_decrease=0.9 - давим шум на 90%
    ref_clean = nr.reduce_noise(y=ref_audio, sr=aligner.sr, prop_decrease=0.9)
    tgt_clean = nr.reduce_noise(y=tgt_audio, sr=aligner.sr, prop_decrease=0.9)

    # Дополнительно: Bandpass фильтр поверх
    ref_clean = aligner.apply_bandpass_filter(ref_clean)
    tgt_clean = aligner.apply_bandpass_filter(tgt_clean)

    # Считаем смещение (Обычная корреляция)
    offset, max_val = aligner.find_offset(ref_clean, tgt_clean, filter_audio=False) # filter уже применили

    print(f"\n>>> Denoised Offset: {offset:.4f}s (Corr: {max_val:.1f})")

    # Визуализация "До" и "После" для одного файла
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.title("Original Target Audio (Dashcam)")
    plt.plot(tgt_audio)
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.title("Denoised Target Audio")
    plt.plot(tgt_clean, color='green')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.title("Correlation Result")
    # Просто покажем текст
    plt.text(0.5, 0.5, f"Found Offset: {offset:.4f}s", ha='center', fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Укажи тут пути к своим файлам
    ref = "data/raw/center_phone/VID_20251027_131854.mp4"
    tgt = "data/raw/left_cam/MOVI0000.avi" # Или MOVI0000.avi

    test_denoising(ref, tgt)
