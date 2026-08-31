from __future__ import annotations

from pathlib import Path
import numpy as np
import librosa
import torch


def load_audio_segment(path: str | Path, sr: int, start_s: float, end_s: float) -> np.ndarray:
    duration = max(0.0, end_s - start_s)
    y, _ = librosa.load(str(path), sr=sr, mono=True, offset=max(0.0, start_s), duration=duration)
    expected = int(round(duration * sr))
    if y.shape[0] < expected:
        y = np.pad(y, (0, expected - y.shape[0]))
    elif y.shape[0] > expected:
        y = y[:expected]
    return y.astype(np.float32, copy=False)


def log_mel(
    y: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 320,
    n_mels: int = 96,
    fmin: float = 30.0,
    fmax: float | None = 7600.0,
) -> torch.Tensor:
    spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=min(fmax or sr / 2, sr / 2),
        power=2.0,
        center=True,
    )
    db = librosa.power_to_db(spec, ref=1.0, top_db=None)
    db = np.clip(db, -80.0, 20.0)
    # Keep a bounded bounded absolute energy scale rather than per song normalization.
    mel = (db + 80.0) / 100.0
    return torch.from_numpy(mel.T.astype(np.float32))  # [frames, mels]
