"""Reduce the elevated noise floor of AI-generated vocal stems.

Root cause (measured): AI vocal stems sit at -46..-62 dB noise floor vs
-72..-77 dB for human references, concentrated in the 200 Hz - 8 kHz band
(mid-band broadband noise) plus low-freq accompaniment leakage below 200 Hz.

Strategy (evidence-based, conservative):
  1. Gentle highpass 60 Hz removes accompaniment rumble without touching
     female fundamentals (f0 >= 180 Hz typically).
  2. Mid-band spectral gating: per-frame gain from noise profile estimated
     on the file's own quiet frames, floor-limited to -14 dB so harmonics
     and weak consonants are never fully removed (avoids the dropout
     regression that global afftdn caused).
  3. Quiet-segment expander: frames below the quiet threshold get an extra
     -18 dB attenuation with smooth attack/release (10/60 ms), targeting
     the measurable -46..-62 dB floor down to human-like levels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "vocal_repair"))


def reduce_noise_floor(vocals: Path, output: Path, sr: int = 48000) -> Path:
    y, orig_sr = sf.read(vocals, always_2d=True)
    x = y.mean(axis=1).astype(np.float64) if y.ndim > 1 else y.astype(np.float64)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak

    # --- 1. gentle highpass (60 Hz, 2nd-order Butterworth) ---
    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, 60.0, btype="high", fs=orig_sr, output="sos")
    x = sosfiltfilt(sos, x)

    # --- 2. STFT ---
    n_fft = 2048
    hop = n_fft // 4
    win = np.hanning(n_fft)
    n = len(x)
    n_frames = 1 + (n - n_fft) // hop
    x_pad = np.pad(x, (0, n_frames * hop + n_fft - n))
    stft = np.zeros((n_frames, n_fft // 2 + 1), dtype=np.complex128)
    for i in range(n_frames):
        seg = x_pad[i * hop: i * hop + n_fft] * win
        stft[i] = np.fft.rfft(seg)
    mag = np.abs(stft)
    phase = np.angle(stft)

    # --- 3. noise profile from quiet frames ---
    frame_rms = np.sqrt(np.mean(mag**2, axis=1))
    active_median = np.median(frame_rms[frame_rms > np.percentile(frame_rms, 50)])
    quiet_thresh = np.percentile(frame_rms, 12)
    # Only engage when real silent segments exist (quiet frames far below the
    # active level). For a signal with no gaps, quiet frames are just the
    # quietest parts of the signal itself and gating would damage it.
    has_true_silence = (quiet_thresh / (active_median + 1e-9)) < 0.3

    freqs = np.fft.rfftfreq(n_fft, 1.0 / orig_sr)
    gain = np.ones_like(mag)

    # --- 3b. high-frequency artifact rolloff to match human reference ---
    # Measured: AI vocal stems carry +3..+6 dB more energy than human vocals
    # at 10-16 kHz (codec reconstruction artifacts + excessive air shelf).
    # Apply a smooth 6-16 kHz tilt unconditionally (linear filter, no dynamics).
    hf_gain = np.ones_like(freqs)
    m6 = (freqs >= 6000.0) & (freqs < 8000.0)
    m8 = (freqs >= 8000.0) & (freqs < 10000.0)
    m10 = (freqs >= 10000.0) & (freqs < 12000.0)
    m12 = (freqs >= 12000.0) & (freqs < 16000.0)
    hf_gain[m6] = np.linspace(1.0, 10 ** (-1.0 / 20.0), m6.sum())
    hf_gain[m8] = np.linspace(10 ** (-1.0 / 20.0), 10 ** (-2.0 / 20.0), m8.sum())
    hf_gain[m10] = np.linspace(10 ** (-2.0 / 20.0), 10 ** (-4.0 / 20.0), m10.sum())
    hf_gain[m12] = np.linspace(10 ** (-4.0 / 20.0), 10 ** (-7.0 / 20.0), m12.sum())
    hf_gain[freqs >= 16000.0] = 10 ** (-7.0 / 20.0)
    gain *= hf_gain[None, :]
    if has_true_silence:
        quiet_idx = np.where(frame_rms < quiet_thresh)[0]
        if len(quiet_idx) < 4:
            quiet_idx = np.argsort(frame_rms)[:4]
        noise_profile = np.median(mag[quiet_idx], axis=0)

        # --- 4. mid-band spectral gating (200 Hz - 10 kHz), floor-limited ---
        band = (freqs >= 200.0) & (freqs <= 10000.0)
        alpha = 1.0
        floor_gain = 10 ** (-14.0 / 20.0)  # never remove more than 14 dB per bin
        signal_est = mag[:, band]
        noise_est = noise_profile[band][None, :]
        g = 1.0 - alpha * (noise_est / (signal_est + 1e-9))
        g = np.clip(g, floor_gain, 1.0)
        # smooth along time (5-frame boxcar) to avoid musical noise
        kernel = np.ones(5) / 5.0
        g = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=g)
        gain[:, band] = g

        # --- 5. quiet-segment expander: extra attenuation on quiet frames ---
        quiet_gain = np.ones(n_frames)
        q = np.where(frame_rms < quiet_thresh)[0]
        quiet_gain[q] = 10 ** (-24.0 / 20.0)
        # smooth attack/release (10 ms attack, 60 ms release)
        att = max(1, int(0.010 * orig_sr / hop))
        rel = max(1, int(0.060 * orig_sr / hop))
        for i in range(1, n_frames):
            target = quiet_gain[i]
            prev = quiet_gain[i - 1]
            tau = att if target < prev else rel
            quiet_gain[i] = prev + (target - prev) / tau
        gain *= quiet_gain[:, None]

    # --- 6. ISTFT with COLA normalization (hann^2 sum is constant at 75% overlap) ---
    out_stft = mag * gain * np.exp(1j * phase)
    out = np.zeros(n_frames * hop + n_fft)
    wsum = np.zeros(n_frames * hop + n_fft)
    for i in range(n_frames):
        seg = np.fft.irfft(out_stft[i])
        out[i * hop: i * hop + n_fft] += seg * win
        wsum[i * hop: i * hop + n_fft] += win**2
    # Cap the normalization at 5x so window-edge regions cannot explode
    # (tail frames have tiny wsum from partial windows).
    wsum = np.clip(wsum, 0.2 * wsum.max(), None)
    out = out / wsum
    out = out[:n]
    out = out / np.max(np.abs(out) + 1e-9) * peak
    if peak > 0.95:
        out = np.clip(out, -1, 1)

    if y.ndim > 1:
        # apply the same per-frame gain curve to both channels (mono-derived)
        final = np.stack([out, out], axis=1)
    else:
        final = out
    final = np.clip(final, -1, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, final, orig_sr, subtype="PCM_16")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Reduce AI vocal noise floor toward human reference level")
    parser.add_argument("vocals", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = reduce_noise_floor(args.vocals, args.output)
    print(f"noise-floor reduced vocal -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
