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

        Args:
            sample_rate: Sampling rate for audio loading.
            22050 Hz is standard for librosa and enough for sync.
        """
        self.sr = sample_rate

    def load_audio(self, video_path: Path, duration: Optional[float] = None) -> NDArray[np.float32]:
        """
        Extracts and loads audio track from a video file.

        Args:
            video_path: Path to the video file.
            duration: Only load the first N seconds (optimization).

        Returns:
            1D numpy array containing the audio time series.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Suppress PySoundFile warnings if any
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # mono=True mixes stereo to mono, which is fine for sync
            y, _ = librosa.load(str(video_path), sr=self.sr, duration=duration, mono=True)

        return y

    def find_offset(self,
                    reference_audio: NDArray[np.float32],
                    target_audio: NDArray[np.float32],
                    max_lag_seconds: float = 30.0) -> Tuple[float, float]:
        """
        Calculates time offset between two audio signals using Cross-Correlation.

        Math:
            Correlation = FFTConvolve(Ref, Target_Reversed)
            Peak implies best overlap.

        Args:
            reference_audio: The 'Master' signal (e.g., Phone).
            target_audio: The signal to shift (e.g., Dashcam).
            max_lag_seconds: Max expected offset to search for (optimization).

        Returns:
            (offset_seconds, max_correlation_value)

            If offset > 0: Target started LATER than Reference.
            If offset < 0: Target started EARLIER than Reference.
        """
        # 1. Normalize signals (Zero-mean, Unit-variance) to handle different volume levels
        ref_norm = (reference_audio - np.mean(reference_audio)) / (np.std(reference_audio) + 1e-9)
        tgt_norm = (target_audio - np.mean(target_audio)) / (np.std(target_audio) + 1e-9)

        # 2. Compute Cross-Correlation via FFT (Fast Fourier Transform)
        # This is O(N log N) vs O(N^2) for standard correlation
        correlation = scipy.signal.fftconvolve(ref_norm, tgt_norm[::-1], mode='full')

        # 3. Find the peak
        lags = np.arange(-len(tgt_norm) + 1, len(ref_norm))
        max_idx = np.argmax(correlation)
        lag_samples = lags[max_idx]

        # 4. Convert samples to seconds
        time_offset = -1 * lag_samples / self.sr
        max_val = correlation[max_idx]

        return time_offset, max_val

# --- Quick Test Block (to run locally) ---
if __name__ == "__main__":
    # Dummy signal generation for testing without files
    print("Running synthetic test...")
    sr = 22050
    # Create a "clap" signal
    base_sig = np.zeros(sr * 10)
    base_sig[sr*2:sr*2+500] = 1 # Clap at 2nd second

    # Target signal is shifted by +1.5 seconds (starts later)
    tgt_sig = np.zeros(sr * 10)
    tgt_sig[sr*3 + int(sr*0.5):sr*3 + int(sr*0.5)+500] = 1 # Clap at 3.5s

    aligner = AudioAligner(sample_rate=sr)
    offset, corr = aligner.find_offset(base_sig, tgt_sig)

    print(f"Calculated Offset: {offset:.4f}s (Expected: ~1.5s)")

    assert np.isclose(offset, 1.5, atol=0.01)
    print("Test Passed!")
