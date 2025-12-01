"""
Module for temporal synchronization of video streams using audio cross-correlation.
"""
import warnings
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import scipy.signal
from numpy.typing import NDArray

class AudioAligner:
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize AudioAligner.
        """
        self.sr = sample_rate

    def load_audio(self, video_path: Path, duration: Optional[float] = None) -> NDArray[np.float32]:
        """Extracts and loads audio track from a video file."""
        if not video_path.exists():
            # Try to handle potential path issues if coming from CSV
            alt_path = Path(str(video_path).strip())
            if not alt_path.exists():
                print(f"Warning: File not found {video_path}")
                return None
            video_path = alt_path

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                y, _ = librosa.load(str(video_path), sr=self.sr, duration=duration, mono=True)
                return y
            except Exception as e:
                print(f"Error reading audio from {video_path.name}: {e}")
                return None

    def apply_bandpass_filter(self, data: np.ndarray, lowcut: float = 300.0, highcut: float = 3000.0, order: int = 5) -> np.ndarray:
        """
        Applies a Butterworth bandpass filter to isolate 'clap' frequencies
        and remove wind/engine noise.
        """
        nyq = 0.5 * self.sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        y = scipy.signal.lfilter(b, a, data)
        return y

    def find_offset(self,
                    reference_audio: NDArray[np.float32],
                    target_audio: NDArray[np.float32],
                    filter_audio: bool = True) -> Tuple[float, float]:
        """
        Calculates time offset between two audio signals.
        """
        if reference_audio is None or target_audio is None:
            return 0.0, 0.0

        # 1. OPTIONAL: Apply Bandpass Filter
        # This is critical for outdoor/marine environments!
        if filter_audio:
            reference_audio = self.apply_bandpass_filter(reference_audio)
            target_audio = self.apply_bandpass_filter(target_audio)

        # 2. Normalize
        ref_norm = (reference_audio - np.mean(reference_audio)) / (np.std(reference_audio) + 1e-9)
        tgt_norm = (target_audio - np.mean(target_audio)) / (np.std(target_audio) + 1e-9)

        # 3. Cross-Correlation
        correlation = scipy.signal.fftconvolve(ref_norm, tgt_norm[::-1], mode='full')

        # 4. Find peak
        lags = np.arange(-len(tgt_norm) + 1, len(ref_norm))
        max_idx = np.argmax(correlation)
        lag_samples = lags[max_idx]

        # 5. Convert to seconds (Inverted sign convention)
        time_offset = -1 * (lag_samples / self.sr)
        max_val = correlation[max_idx]

        return time_offset, max_val

if __name__ == "__main__":
    # Simple self-test
    aligner = AudioAligner()
    # Create synthetic signal with noise
    sr = 22050
    t = np.linspace(0, 10, sr*10)

    # Signal 1: Noise + Clap at 2s
    sig1 = np.sin(2 * np.pi * 50 * t) * 0.1 # Low freq noise (50Hz)
    sig1[sr*2:sr*2+500] += 5.0 # Sharp clap

    # Signal 2: Noise + Clap at 3.5s (Offset = +1.5s)
    sig2 = np.sin(2 * np.pi * 50 * t) * 0.1
    sig2[sr*3 + int(sr*0.5):sr*3 + int(sr*0.5)+500] += 5.0

    offset, _ = aligner.find_offset(sig1, sig2, filter_audio=True)
    print(f"Test Offset (Filtered): {offset:.4f}s (Expected 1.5s)")
    assert np.isclose(offset, 1.5, atol=0.05)
