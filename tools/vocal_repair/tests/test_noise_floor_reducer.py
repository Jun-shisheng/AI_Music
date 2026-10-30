import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noise_floor_reducer import reduce_noise_floor  # noqa: E402


def _noisy_signal(sr: int = 48000, seconds: float = 1.2) -> tuple[np.ndarray, float]:
    """0.3 s pure noise + 0.9 s sine + noise; returns signal and noise floor."""
    n = int(sr * seconds)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1.0, n)
    noise = noise / np.max(np.abs(noise)) * 10 ** (-40.0 / 20.0)
    t = np.arange(n) / sr
    sine = 0.6 * np.sin(2 * np.pi * 220 * t)
    x = noise.copy()
    x[int(0.3 * sr):] += sine[int(0.3 * sr):]
    floor = 20 * np.log10(np.sqrt(np.mean(noise[: int(0.3 * sr)] ** 2)) + 1e-12)
    return x, floor


def _quiet_floor_db(path: Path) -> float:
    y, sr = sf.read(path, always_2d=True)
    x = y.mean(axis=1)
    frame_len = int(sr * 0.05)
    n_frames = len(x) // frame_len
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    quiet = np.percentile(rms, 10)
    return float(20 * np.log10(quiet + 1e-12))


def test_reduce_noise_floor_lowers_floor_and_keeps_sine(tmp_path):
    x, _ = _noisy_signal()
    src = tmp_path / "noisy.wav"
    sf.write(src, np.stack([x, x], axis=1), 48000, subtype="PCM_16")
    before = _quiet_floor_db(src)

    out = tmp_path / "reduced.wav"
    reduce_noise_floor(src, out)

    after = _quiet_floor_db(out)
    assert after < before - 10.0, f"noise floor not lowered: {before:.1f} -> {after:.1f} dB"
    assert after < -60.0, f"floor still too high: {after:.1f} dB"

    y, _ = sf.read(out, always_2d=True)
    mono = y.mean(axis=1)
    t = np.arange(len(mono)) / 48000
    # sine energy at 220 Hz must survive (correlation with reference sine)
    ref = np.sin(2 * np.pi * 220 * t)
    voiced = mono[int(0.35 * 48000):]
    corr = np.abs(np.corrcoef(voiced, ref[int(0.35 * 48000):])[0, 1])
    assert corr > 0.9, f"sine damaged by noise reduction: corr={corr:.3f}"


def test_reduce_noise_floor_is_idempotent_on_clean_signal(tmp_path):
    """A clean sine (no added noise) must pass through nearly unchanged."""
    sr = 48000
    t = np.arange(int(sr * 0.6)) / sr
    clean = 0.5 * np.sin(2 * np.pi * 330 * t)
    src = tmp_path / "clean.wav"
    sf.write(src, np.stack([clean, clean], axis=1), sr, subtype="PCM_16")

    out = tmp_path / "clean_out.wav"
    reduce_noise_floor(src, out)

    y, _ = sf.read(out, always_2d=True)
    mono = y.mean(axis=1)[: len(clean)]
    corr = np.abs(np.corrcoef(mono, clean)[0, 1])
    assert corr > 0.97, f"clean signal altered: corr={corr:.3f}"
