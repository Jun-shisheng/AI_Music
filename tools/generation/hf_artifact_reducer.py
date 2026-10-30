"""Surgical 8-16k artifact reduction on AI vocals (nasal/stress positions).

Design (distinct from the abandoned repair chain):
- Per-frame RELATIVE gate: a frame is treated only when its 8-16k energy is
  abnormally high RELATIVE TO ITS OWN mid-band (300-3000 Hz). This catches
  codec artifacts that sit on top of nasal/stress phonemes while leaving normal
  frames (which have a normal high/mid ratio) completely untouched.
- The reduction is also RELATIVE: the frame's 8-16k band is scaled toward its
  own mid-band ratio (median of the whole file), so a genuinely bright frame is
  only slightly tamed and never made dull.
- Below 8 kHz is never touched; the mid/voice band keeps its exact spectrum.

This is the minimal change that addresses the measured artifact signature
(AI song carries +4.9/+6.0 dB at 8-12k/12-16k vs human refs, clustering on
voiced frames), without a global high-shelf that would eat sibilance/air.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "vocal_repair"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

N_FFT = 2048
HOP = 512


def reduce_hf_artifacts(input_wav: Path, output_wav: Path,
                        ratio_thresh: float = 3.0,
                        max_reduction_db: float = -8.0) -> dict:
    import librosa

    y, sr = sf.read(input_wav, always_2d=True)
    # Detect abnormal frames from a mono blend, but reconstruct every channel
    # independently so stereo width and accompaniment imaging are preserved.
    mono = y.mean(axis=1).astype(np.float64)
    S = librosa.stft(mono, n_fft=N_FFT, hop_length=HOP, center=True)
    mag = np.abs(S)
    phase = np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    b_hi = (freqs >= 8000) & (freqs <= 16000)
    b_mid = (freqs >= 300) & (freqs < 3000)

    hi_power = mag[b_hi] ** 2
    mid_power = mag[b_mid] ** 2
    hi_energy = hi_power.mean(axis=0)
    mid_energy = mid_power.mean(axis=0)

    ratio = hi_energy / (mid_energy + 1e-9)
    med_ratio = float(np.median(ratio))

    # frames with abnormal ratio (and some mid content so we don't touch silence)
    active = mid_energy > np.percentile(mid_energy, 30)
    targets = np.where(active & (ratio > med_ratio * ratio_thresh))[0]

    if len(targets) == 0:
        sf.write(output_wav, y, sr, subtype="PCM_16")
        return {"frames": 0, "med_ratio": med_ratio, "max_reduction_db": 0.0,
                "channels_preserved": int(y.shape[1])}

    # per-target gain: scale hi band so its ratio returns to ~med_ratio, capped
    gains = np.ones_like(mag)
    gain_db = np.zeros(len(ratio))
    for i in targets:
        current_db = 10 * np.log10(ratio[i] / med_ratio)  # how many dB above normal
        cut_db = min(0.0, max(max_reduction_db, -current_db))
        gain_db[i] = cut_db
        gains[b_hi, i] = 10 ** (cut_db / 20)

    # temporal smoothing (10-frame boxcar) so edits fade instead of click
    kernel = np.ones(10) / 10.0
    smooth = np.convolve(gain_db, kernel, mode="same")
    smooth_gain = np.ones_like(mag)
    for i in range(len(smooth)):
        if smooth[i] < 0:
            smooth_gain[b_hi, i] = 10 ** (smooth[i] / 20)

    repaired_channels = []
    for channel in range(y.shape[1]):
        channel_stft = librosa.stft(y[:, channel].astype(np.float64),
                                    n_fft=N_FFT, hop_length=HOP, center=True)
        channel_mag = np.abs(channel_stft)
        channel_phase = np.angle(channel_stft)
        channel_out = channel_mag * smooth_gain * np.exp(1j * channel_phase)
        repaired_channels.append(librosa.istft(
            channel_out, hop_length=HOP, center=True, length=len(y)))
    out = np.stack(repaired_channels, axis=1)
    out = np.clip(out, -1, 1)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, out, sr, subtype="PCM_16")
    return {
        "frames": int(len(targets)),
        "fraction": round(len(targets) / len(ratio), 3),
        "med_ratio": round(med_ratio, 4),
        "max_reduction_db": round(float(np.min(gain_db[targets])), 2),
        "mean_reduction_db": round(float(np.mean(gain_db[targets])), 2),
        "channels_preserved": int(y.shape[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Surgical 8-16k artifact reduction")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ratio-thresh", type=float, default=3.0)
    parser.add_argument("--max-cut-db", type=float, default=-8.0)
    args = parser.parse_args()
    result = reduce_hf_artifacts(args.input, args.output,
                                 args.ratio_thresh, args.max_cut_db)
    print(result)
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
