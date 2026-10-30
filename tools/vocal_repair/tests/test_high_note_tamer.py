import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from high_note_tamer import _f0_frames, tame_high_notes  # noqa: E402


def _high_note_signal(sr: int = 44100, seconds: float = 1.0, f0: float = 440.0):
    """High-pitch vocal-like signal: strong formant at 2.2 kHz over a 440 Hz f0."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    # fundamental + 2nd..8th harmonics, 5th harmonic (2200 Hz) emphasized
    x = np.zeros(n)
    for k in range(1, 9):
        amp = 1.0 / k
        if k == 5:
            amp *= 3.0  # harsh formant peak
        x += amp * np.sin(2 * np.pi * k * f0 * t)
    x = x / np.max(np.abs(x))
    return x


def _band_energy_db(path: Path, lo: float, hi: float) -> float:
    import librosa
    y, sr = sf.read(path, always_2d=True)
    x = y.mean(axis=1)
    x = x / np.max(np.abs(x))
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= lo) & (freqs < hi)
    return float(10 * np.log10(S[band].mean() + 1e-12))


def test_tame_high_notes_reduces_formant_peak(tmp_path):
    x = _high_note_signal()
    src = tmp_path / "high.wav"
    sf.write(src, np.stack([x, x], axis=1), 44100, subtype="PCM_16")

    out = tmp_path / "tamed.wav"
    tame_high_notes(src, out, f0_thresh=380.0, max_cut_db=-8.0)

    before = _band_energy_db(src, 2000, 4000)
    after = _band_energy_db(out, 2000, 4000)
    # the 2-4 kHz formant region (harshness zone) must drop meaningfully
    assert after < before - 4.0, f"formant band not reduced: {before:.1f} -> {after:.1f} dB"
    # while the fundamental survives (voice stays intact)
    assert _band_energy_db(out, 300, 700) > _band_energy_db(out, 2000, 4000) - 20


def test_tame_high_notes_leaves_low_pitch_untouched(tmp_path):
    sr = 44100
    t = np.arange(int(sr * 0.8)) / sr
    x = 0.6 * np.sin(2 * np.pi * 220 * t)
    src = tmp_path / "low.wav"
    sf.write(src, np.stack([x, x], axis=1), sr, subtype="PCM_16")

    out = tmp_path / "low_tamed.wav"
    tame_high_notes(src, out)

    y, _ = sf.read(out, always_2d=True)
    mono = y.mean(axis=1)[: len(x)]
    corr = np.abs(np.corrcoef(mono, x)[0, 1])
    assert corr > 0.95, f"low pitch altered: corr={corr:.3f}"


def test_f0_frames_detects_high_notes():
    sr = 44100
    t = np.arange(int(sr * 0.5)) / sr
    x = 0.5 * np.sin(2 * np.pi * 480 * t)
    f0 = _f0_frames(x, sr, 512, 2048)
    voiced = f0[f0 > 0]
    assert len(voiced) > 0
    assert np.median(voiced) > 400, f"f0 not detected at 480Hz: {np.median(voiced):.0f}"
