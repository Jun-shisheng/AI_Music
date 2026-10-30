"""Diagnose noise and discontinuity defects in generated audio.

Usage:
    python diagnose_audio.py <wav1> [wav2 ...]

Detects:
  - clicks (sample-level jumps)
  - dropouts (short-window RMS collapses between active frames)
  - spectral discontinuities (frame-to-frame spectrum jumps)
  - high-frequency noise floor (8-16kHz energy during low-activity frames)
  - tremolo / amplitude modulation in mid-band
  - stereo channel correlation anomalies
  - DC offset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def analyze(path: Path) -> dict:
    audio, sr = sf.read(path, always_2d=True)
    if audio.shape[1] >= 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio[:, 0]
    if mono.dtype != np.float64:
        mono = mono.astype(np.float64)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak > 0:
        mono = mono / peak
    n = len(mono)
    dur = n / sr

    report: dict = {
        "file": str(path),
        "sr": sr,
        "channels": audio.shape[1],
        "duration_s": round(dur, 3),
        "peak": peak,
    }

    # --- DC offset ---
    report["dc_offset"] = float(np.mean(mono))

    # --- RMS profile at 10ms frames ---
    frame_len = int(sr * 0.01)
    n_frames = n // frame_len
    frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames**2, axis=1))

    # --- clicks: sample-to-sample jumps > threshold ---
    diff = np.abs(np.diff(mono))
    click_thresh = 0.25
    click_idx = np.where(diff > click_thresh)[0]
    clicks = []
    if len(click_idx):
        # merge clicks within 10ms into one event
        groups = np.split(click_idx, np.where(np.diff(click_idx) > sr * 0.01)[0] + 1)
        for g in groups:
            clicks.append({
                "t": round(float(g[0] / sr), 3),
                "max_jump": round(float(np.max(diff[g])), 3),
            })
    report["clicks"] = clicks
    report["click_count"] = len(clicks)

    # --- dropouts: 10ms RMS collapse between active frames ---
    active_floor = max(0.02, float(np.percentile(rms, 35)) * 0.7)
    dropout_events = []
    i = 0
    while i < n_frames:
        prev_active = rms[max(0, i - 3):i]
        next_active = rms[i + 1:i + 4]
        prev_ok = len(prev_active) and np.mean(prev_active) > active_floor
        next_ok = len(next_active) and np.mean(next_active) > active_floor
        if rms[i] < active_floor * 0.25 and prev_ok and next_ok:
            j = i
            while j < n_frames and rms[j] < active_floor * 0.25:
                j += 1
            dropout_events.append({
                "start": round(i * 0.01, 3),
                "end": round(j * 0.01, 3),
                "len_s": round((j - i) * 0.01, 3),
                "depth": round(float(1.0 - rms[i:j].max() / max(active_floor, 1e-9)), 3),
            })
            i = j
        i += 1
    report["dropouts"] = dropout_events
    report["dropout_total_s"] = round(sum(e["len_s"] for e in dropout_events), 3)

    # --- spectral analysis ---
    import scipy.signal as sig

    nperseg = 2048
    f, t, Sxx = sig.spectrogram(mono, sr, nperseg=nperseg, noverlap=nperseg // 2)
    Sxx = Sxx + 1e-12
    Sxx_db = 10 * np.log10(Sxx)

    # frame-level loudness from spectrogram for "low activity" frames
    frame_rms_db = 10 * np.log10(np.mean(Sxx, axis=0) + 1e-12)

    # high-frequency noise floor: median 8-16kHz energy in low-activity frames
    hf_mask = (f >= 8000) & (f <= 16000)
    low_act = frame_rms_db < (np.percentile(frame_rms_db, 35) - 3)
    if np.any(low_act):
        hf_floor = 10 * np.log10(np.median(Sxx[hf_mask][:, low_act]) + 1e-12)
        hf_floor_active = 10 * np.log10(np.median(Sxx[hf_mask][:, ~low_act]) + 1e-12)
    else:
        hf_floor = hf_floor_active = None
    report["hf_noise_floor_db"] = round(float(hf_floor), 2) if hf_floor is not None else None
    report["hf_noise_floor_active_db"] = round(float(hf_floor_active), 2) if hf_floor_active is not None else None
    # hf ratio overall
    hf_energy = float(np.sum(Sxx[hf_mask]))
    total_energy = float(np.sum(Sxx))
    report["hf_energy_ratio"] = round(hf_energy / max(total_energy, 1e-12), 4)

    # broadband noise detection: frames where many bands are simultaneously elevated
    # (noise-like) vs sparse (tonal)
    band_active = Sxx_db > (np.percentile(Sxx_db, 60, axis=0)[None, :])
    active_fraction = np.mean(band_active, axis=0)
    noisy_frames = active_fraction > 0.6
    report["noise_like_frame_ratio"] = round(float(np.mean(noisy_frames)), 4)

    # --- spectral discontinuity: large frame-to-frame spectrum change ---
    spec_norm = Sxx / (np.sum(Sxx, axis=0, keepdims=True) + 1e-12)
    spec_diff = np.linalg.norm(np.diff(spec_norm, axis=1), axis=0)
    spec_diff_db = 10 * np.log10(spec_diff + 1e-12)
    disc_thresh = np.percentile(spec_diff_db, 97) if len(spec_diff_db) else 0
    disc_events = []
    if len(spec_diff_db):
        idx = np.where(spec_diff_db > max(disc_thresh, -20))[0]
        if len(idx):
            groups = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
            for g in groups:
                # hop = nperseg/2 = 1024 samples
                mid = (g[0] + g[-1]) // 2
                disc_events.append({
                    "t": round(float(t[mid]), 3),
                    "jump_db": round(float(spec_diff_db[mid]), 2),
                })
    report["spectral_discontinuities"] = disc_events

    # --- tremolo: envelope modulation of 300-3000Hz band ---
    band = (f >= 300) & (f <= 3000)
    band_env = 10 * np.log10(np.sum(Sxx[band], axis=0) + 1e-12)
    band_env_detrend = band_env - np.convolve(band_env, np.ones(40) / 40, mode="same")
    mod = np.abs(np.diff(band_env_detrend))
    report["midband_modulation_flux"] = round(float(np.percentile(mod, 95)), 4)

    # --- stereo correlation ---
    if audio.shape[1] >= 2:
        l = audio[:, 0]
        r = audio[:, 1]
        corr = np.corrcoef(l, r)[0, 1]
        report["stereo_corr"] = round(float(corr), 4)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    reports = [analyze(p) for p in args.files]
    for r in reports:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
