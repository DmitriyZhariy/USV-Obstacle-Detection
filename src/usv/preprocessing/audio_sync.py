"""
Module for temporal synchronization of video streams.
Features:
- Standard Cross-Correlation (Robust)
- Spectral Gating Denoising (noisereduce)
- Aggressive High-Pass Filtering (to isolate claps from engine rumble)
- Native 16kHz processing
"""
import warnings
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import scipy.signal
import noisereduce as nr
from numpy.typing import NDArray

class AudioAligner:
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

    def load_audio(self, video_path: Path, duration: Optional[float] = None) -> NDArray[np.float32]:
        """Extracts and loads audio track from a video file."""
        if not video_path.exists():
            video_path = Path(str(video_path))
            if not video_path.exists():
                return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Load as mono
                y, _ = librosa.load(str(video_path), sr=self.sr, duration=duration, mono=True)
                return y
            except Exception as e:
                print(f"Error loading audio {video_path}: {e}")
                return None

    def clean_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Applies cleaning chain: Denoise -> High-Pass Filter.
        """
        if len(data) == 0: return data

        # 1. Spectral Gating (Remove constant background noise)
        try:
            # Less aggressive prop_decrease to keep signal integrity
            data_clean = nr.reduce_noise(y=data, sr=self.sr, prop_decrease=0.7, n_fft=512)
        except Exception:
            data_clean = data

        # 2. Aggressive High-Pass / Bandpass Filter
        # CRITICAL FIX: We moved from 300Hz to 2500Hz to remove structural vibration (thuds).
        # We target the sharp "click" of the clap/buttons.
        nyq = 0.5 * self.sr
        low = 2500.0 / nyq  # Was 300.0
        high = 7500.0 / nyq # Was 4000.0

        # Safety check for Nyquist
        if high >= 1.0: high = 0.99

        b, a = scipy.signal.butter(5, [low, high], btype='band')
        data_filtered = scipy.signal.lfilter(b, a, data_clean)

        return data_filtered

    def find_offset(self,
                    ref_sig: NDArray[np.float32],
                    tgt_sig: NDArray[np.float32],
                    apply_cleaning: bool = True) -> Tuple[float, float]:
        """
        Calculates time offset.
        """
        if ref_sig is None or tgt_sig is None:
            return 0.0, 0.0

        if apply_cleaning:
            ref_sig = self.clean_signal(ref_sig)
            tgt_sig = self.clean_signal(tgt_sig)

        # Normalize (Z-score)
        ref_norm = (ref_sig - np.mean(ref_sig)) / (np.std(ref_sig) + 1e-9)
        tgt_norm = (tgt_sig - np.mean(tgt_sig)) / (np.std(tgt_sig) + 1e-9)

        # Correlation
        correlation = scipy.signal.fftconvolve(ref_norm, tgt_norm[::-1], mode='full')

        lags = np.arange(-len(tgt_norm) + 1, len(ref_norm))
        max_idx = np.argmax(correlation)
        lag_samples = lags[max_idx]

        # Offset calculation
        time_offset = lag_samples / self.sr
        max_val = correlation[max_idx]

        return time_offset, max_val

if __name__ == "__main__":
    print("Fixed AudioAligner ready (High-Pass > 2500Hz).")
