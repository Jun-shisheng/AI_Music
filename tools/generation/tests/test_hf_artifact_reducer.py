"""Tests for the surgical HF-artifact reducer (hf_artifact_reducer.py).

Key guarantees:
- A clean signal (normal high/mid ratio everywhere) is untouched.
- A signal with injected 8-16k bursts on top of a voiced tone has those bursts
  reduced, while the mid band (voice) is preserved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hf_artifact_reducer import reduce_hf_artifacts  # noqa: E402

SR = 48000
N_FFT = 2048
HOP = 512


def _band_db(path: Path, lo: float, hi: float) -> float:
    import librosa

    y, sr = sf.read(path, always_2d=True)
    mono = y.mean(axis=1).astype(np.float64)
    S = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    m = (freqs >= lo) & (freqs < hi)
    return float(10 * np.log10(S[m].mean() + 1e-12))


def _write_signal(tmp_path, name: str, with_bursts: bool) -> Path:
    """2s voice-like tone (300-3000 content) with optional 8-16k bursts.

    The bursts are added ON TOP of an already-strong voiced section (mid stays
    high and HF is relatively stronger) — mimicking nasal/stress phonemes where
    the codec adds HF artifacts while the voiced core remains present.
    """
    sr = SR
    t = np.arange(int(sr * 2)) / sr
    x = (0.4 * np.sin(2 * np.pi * 220 * t)
         + 0.25 * np.sin(2 * np.pi * 440 * t)
         + 0.15 * np.sin(2 * np.pi * 880 * t)
         + 0.08 * np.sin(2 * np.pi * 1760 * t))
    rng = np.random.default_rng(7)
    if with_bursts:
        from scipy import signal as sig

        noise = rng.normal(0, 1, len(t))
        sos = sig.butter(4, [8000 / (sr / 2), 16000 / (sr / 2)], btype="bandpass", output="sos")
        hf = sig.sosfilt(sos, noise)
        # add HF bursts in windows where the tone is already near its peak so
        # mid stays active and the high/mid ratio spikes
        boost = np.zeros(len(t))
        boost[int(0.4 * sr):int(0.55 * sr)] = 1.0
        boost[int(1.2 * sr):int(1.35 * sr)] = 1.0
        x = x + hf * boost * 1.8
        # keep the tone present under the burst (mid remains strong)
        x = x + 0.3 * np.sin(2 * np.pi * 440 * t) * boost
    p = tmp_path / name
    sf.write(p, np.stack([x, x], axis=1), sr, subtype="PCM_16")
    return p


def test_clean_signal_untouched(tmp_path):
    src = _write_signal(tmp_path, "clean.wav", with_bursts=False)
    out = tmp_path / "out.wav"
    reduce_hf_artifacts(src, out, ratio_thresh=3.0, max_reduction_db=-8.0)
    y, _ = sf.read(out, always_2d=True)
    src_y, _ = sf.read(src, always_2d=True)
    # clean signal should be nearly identical (no frames flagged)
    assert np.max(np.abs(y - src_y)) < 0.02


def test_bursts_reduced_mid_preserved(tmp_path):
    src = _write_signal(tmp_path, "bursty.wav", with_bursts=True)
    out = tmp_path / "out.wav"
    result = reduce_hf_artifacts(src, out, ratio_thresh=3.0, max_reduction_db=-8.0)
    assert result["frames"] > 0, "no frames flagged on bursty signal"

    # HF burst band reduced
    assert _band_db(out, 8000, 16000) < _band_db(src, 8000, 16000) - 2.0
    # mid voice band preserved within 1 dB
    assert abs(_band_db(out, 300, 3000) - _band_db(src, 300, 3000)) < 1.0


def test_output_bounded(tmp_path):
    src = _write_signal(tmp_path, "b.wav", with_bursts=True)
    out = tmp_path / "out.wav"
    reduce_hf_artifacts(src, out)
    y, _ = sf.read(out, always_2d=True)
    assert float(np.max(np.abs(y))) <= 1.0


def test_stereo_image_is_not_folded_to_mono(tmp_path):
    src = tmp_path / "stereo.wav"
    t = np.arange(SR * 2) / SR
    left = 0.3 * np.sin(2 * np.pi * 220 * t)
    right = 0.3 * np.sin(2 * np.pi * 330 * t)
    sf.write(src, np.stack([left, right], axis=1), SR, subtype="PCM_16")
    out = tmp_path / "stereo_out.wav"
    reduce_hf_artifacts(src, out)
    y, _ = sf.read(out, always_2d=True)
    assert y.shape[1] == 2
    assert not np.allclose(y[:, 0], y[:, 1], atol=0.01)
