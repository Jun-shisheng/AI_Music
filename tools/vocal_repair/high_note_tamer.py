"""Tame harsh high-note formants in AI vocal stems.

Root cause (measured): at high pitch (F0 > 380 Hz) the AI stem carries a
very strong narrow resonance (~2.2 kHz, peak-band gap 19.9 dB) vs human
references (gap ~11.8 dB). This reads as shouting/electro harshness on
sustained high notes.

Fix: only frames with F0 > threshold get a gentle adaptive notch at their
strongest 1.5-4 kHz peak, limited to -5 dB, with time smoothing so the
treatment fades in/out of the phrase instead of clicking.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "vocal_repair"))


def _f0_frames(x: np.ndarray, sr: int, hop: int, n_fft: int) -> np.ndarray:
    """Estimate F0 per STFT frame via pyworld (robust against octave slips)."""
    import pyworld as pw
    n_frames = 1 + (len(x) - n_fft) // hop
    frame_period = hop * 1000.0 / sr
    f0_raw, _ = pw.harvest(x.astype(np.float64), sr, frame_period=frame_period)
    f0 = np.zeros(n_frames)
    k = min(len(f0_raw), n_frames)
    f0[:k] = f0_raw[:k]
    # reject isolated octave slips: a frame far from its 5-frame neighborhood
    # median while the neighbors agree with each other is spurious
    from scipy.ndimage import median_filter
    voiced = f0 > 0
    if voiced.sum() > 6:
        med = median_filter(f0, size=5, mode="nearest")
        for i in np.where(voiced)[0]:
            nb = [f0[j] for j in range(max(0, i - 2), min(n_frames, i + 3))
                  if j != i and f0[j] > 0]
            if len(nb) >= 3:
                nb_med = float(np.median(nb))
                if abs(12 * np.log2(f0[i] / (nb_med + 1e-9))) > 4.5:
                    f0[i] = 0.0
    return f0


def tame_high_notes(vocals: Path, output: Path, f0_thresh: float = 350.0,
                    max_cut_db: float = -10.0) -> Path:
    y, sr = sf.read(vocals, always_2d=True)
    x = y.mean(axis=1).astype(np.float64) if y.ndim > 1 else y.astype(np.float64)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak

    n_fft = 2048
    hop = n_fft // 4
    win = np.hanning(n_fft)
    n = len(x)
    n_frames = 1 + (n - n_fft) // hop
    x_pad = np.pad(x, (0, n_frames * hop + n_fft - n))
    stft = np.zeros((n_frames, n_fft // 2 + 1), dtype=np.complex128)
    for i in range(n_frames):
        stft[i] = np.fft.rfft(x_pad[i * hop: i * hop + n_fft] * win)
    mag = np.abs(stft)
    phase = np.angle(stft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    f0 = _f0_frames(x, sr, hop, n_fft)
    high_frames = f0 > f0_thresh

    if high_frames.any():
        # find strongest 1.5-4 kHz peak per high frame (per-frame; temporal
        # smoothing comes from the strength fade below, not the peak position)
        band = (freqs >= 1500.0) & (freqs <= 4000.0)
        band_freqs = freqs[band]
        peak_freq = np.zeros(n_frames)
        for i in np.where(high_frames)[0]:
            peak_freq[i] = band_freqs[np.argmax(mag[i, band])]

        # per-bin gain: wide 1.5-4 kHz shelf (-5 dB) so the formant region sits
        # closer to the human gap (~12 dB), plus a deeper single-peak notch.
        gain = np.ones_like(mag)
        shelf_db = -5.0
        gain[:, band] *= 10 ** (shelf_db / 20.0)
        depth = 10 ** (max_cut_db / 20.0)
        for i in np.where(high_frames)[0]:
            pf = peak_freq[i]
            if pf <= 0:
                continue
            spread = 0.4  # octave width of the notch (wider = gentler, less surgical)
            ratio = np.where(freqs > 0, freqs / pf, 1.0)
            notch = 1.0 - (1.0 - depth) * np.exp(-((np.log2(ratio)) ** 2) / (2 * spread**2))
            gain[i] *= notch
        # temporal fade-in/out near phrase edges (attack 40ms, release 80ms)
        frames_n = len(gain)
        att = max(1, int(0.040 * sr / hop))
        rel = max(1, int(0.080 * sr / hop))
        strength = np.where(high_frames, 1.0, 0.0)
        for i in range(1, frames_n):
            target = strength[i]
            prev = strength[i - 1]
            tau = att if target > prev else rel
            strength[i] = prev + (target - prev) / tau
        gain = 1.0 - strength[:, None] * (1.0 - gain)

        out_stft = mag * gain * np.exp(1j * phase)
    else:
        out_stft = stft

    out = np.zeros(n_frames * hop + n_fft)
    wsum = np.zeros(n_frames * hop + n_fft)
    for i in range(n_frames):
        seg = np.fft.irfft(out_stft[i])
        out[i * hop: i * hop + n_fft] += seg * win
        wsum[i * hop: i * hop + n_fft] += win**2
    wsum = np.clip(wsum, 0.2 * wsum.max(), None)
    out = out / wsum
    out = out[:n]
    # input was normalized to `peak`; scale back by the *input* peak so a
    # formant cut is a real cut (no renormalization masking the reduction)
    out = out * peak
    out = np.clip(out, -1, 1)

    if y.ndim > 1:
        final = np.stack([out, out], axis=1)
    else:
        final = out
    final = np.clip(final, -1, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, final, sr, subtype="PCM_16")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Tame harsh high-note formants")
    parser.add_argument("vocals", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--f0-thresh", type=float, default=380.0)
    parser.add_argument("--max-cut-db", type=float, default=-5.0)
    args = parser.parse_args()
    tame_high_notes(args.vocals, args.output, args.f0_thresh, args.max_cut_db)
    print(f"tamed high-note vocal -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
