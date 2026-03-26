import os
from typing import Optional, Tuple

import numpy as np
import librosa


def load_audio(
    path: str,
    sr: int = 16000,
    mono: bool = True,
) -> np.ndarray:
    """
    Load an audio file and resample it.

    Args:
        path: Path to .wav file
        sr: Target sample rate
        mono: Convert audio to mono if True

    Returns:
        audio: 1D numpy array of float32 samples
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, _ = librosa.load(path, sr=sr, mono=mono)

    if audio.size == 0:
        raise ValueError(f"Loaded empty audio file: {path}")

    return audio.astype(np.float32)


def audio_to_logmel(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 64,
    n_fft: int = 400,
    hop_length: int = 160,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    Convert waveform to log-mel spectrogram.

    Args:
        audio: 1D float waveform
        sr: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Hop size between frames
        fmin: Minimum frequency
        fmax: Maximum frequency (None -> sr/2)

    Returns:
        logmel: shape (n_mels, time_frames)
    """
    if fmax is None:
        fmax = sr / 2

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel.astype(np.float32)


def fix_time_dimension(
    spec: np.ndarray,
    target_frames: int,
    pad_value: Optional[float] = None,
) -> np.ndarray:
    """
    Pad or crop spectrogram to a fixed number of time frames.

    Args:
        spec: Spectrogram of shape (n_mels, time_frames)
        target_frames: Desired number of time frames
        pad_value: Value to pad with. If None, uses spec.min()

    Returns:
        fixed_spec: shape (n_mels, target_frames)
    """
    if spec.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {spec.shape}")

    n_mels, time_frames = spec.shape

    if time_frames == target_frames:
        return spec

    if time_frames > target_frames:
        return spec[:, :target_frames]

    if pad_value is None:
        pad_value = float(spec.min())

    pad_width = target_frames - time_frames
    padded = np.pad(
        spec,
        pad_width=((0, 0), (0, pad_width)),
        mode="constant",
        constant_values=pad_value,
    )
    return padded.astype(np.float32)


def normalize_spectrogram(
    spec: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize spectrogram.

    If mean/std are not provided, normalize per sample.

    Args:
        spec: Spectrogram array
        mean: Dataset mean (optional)
        std: Dataset std (optional)
        eps: Small value to avoid divide-by-zero

    Returns:
        normalized spectrogram
    """
    if mean is None or std is None:
        mean = float(np.mean(spec))
        std = float(np.std(spec))

    return ((spec - mean) / (std + eps)).astype(np.float32)


def wav_to_logmel(
    path: str,
    sr: int = 16000,
    n_mels: int = 64,
    n_fft: int = 400,
    hop_length: int = 160,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
    target_frames: int = 101,
    normalize: bool = True,
) -> np.ndarray:
    """
    Full pipeline:
    wav -> waveform -> log-mel -> fixed length -> optional normalize -> add channel dim

    Returns:
        spec: shape (n_mels, target_frames, 1)
    """
    audio = load_audio(path, sr=sr, mono=True)

    spec = audio_to_logmel(
        audio=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )

    spec = fix_time_dimension(spec, target_frames=target_frames)

    if normalize:
        spec = normalize_spectrogram(spec)

    spec = np.expand_dims(spec, axis=-1)  # (n_mels, time_frames, 1)
    return spec.astype(np.float32)


def get_label_from_path(path: str) -> int:
    """
    Example label helper:
    expects folder name to be either 'click' or 'no_click'

    Returns:
        1 for click
        0 for no_click
    """
    folder = os.path.basename(os.path.dirname(path)).lower()

    if folder == "click":
        return 1
    if folder == "no_click":
        return 0

    raise ValueError(
        f"Could not infer label from parent folder '{folder}'. "
        f"Expected 'click' or 'no_click'."
    )


def process_file(
    path: str,
    sr: int = 16000,
    n_mels: int = 64,
    n_fft: int = 400,
    hop_length: int = 160,
    target_frames: int = 101,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Convenience function for one file.

    Returns:
        x: shape (n_mels, target_frames, 1)
        y: integer label
    """
    x = wav_to_logmel(
        path=path,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        target_frames=target_frames,
        normalize=normalize,
    )
    y = get_label_from_path(path)
    return x, y