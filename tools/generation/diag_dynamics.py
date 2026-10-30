"""Diagnostic: how wide is a take's dynamic / pitch / rhythm range?

Written 2026-10-15 to turn three listening complaints into numbers before
touching the generator:

  1. "低的时候不够低，高的时候不够高，有点平了" -- the quiet parts are not soft
     enough and the loud parts are not big enough: intra-song contrast.
  2. "冲击感/神曲感" -- does the climax actually hit (low-end weight, transient
     density), and does the song groove (onset density / beat strength)?
  3. "前奏太长" -- needs the vocal onset time, which whisper gives directly
     (see diction_qa.py --first-vocal).

This is a measurement tool only; nothing here feeds generation.

Usage:
  python tools/generation/diag_dynamics.py --dir outputs/2026-10-14_epic_en/selected
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def _db(x: np.ndarray, ref: float) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, 1e-12) / max(ref, 1e-12))


def measure(path: Path, with_f0: bool = True) -> dict:
    import librosa

    y, sr = sf.read(path, always_2d=True)
    mono = y.mean(axis=1).astype(np.float32)
    dur = len(mono) / sr

    # --- level envelope -----------------------------------------------------
    win = int(sr)
    n = max(len(mono) // win, 1)
    rms = np.sqrt(np.maximum((mono[: n * win].reshape(n, win) ** 2).mean(1), 1e-20))
    peak = float(rms.max())
    db = _db(rms, peak)

    p95, p50, p10 = (float(np.percentile(db, q)) for q in (95, 50, 10))

    # quietest sustained passage (4 s) and loudest 2 s, both as 0-peak dB
    def _extremum(width_s: int, mode: str) -> tuple[float, float]:
        w = max(int(width_s), 1)
        if w >= len(rms):
            return float(db.mean()), 0.0
        sm = np.convolve(db, np.ones(w) / w, mode="valid")
        i = int(np.argmin(sm) if mode == "min" else np.argmax(sm)) + w // 2
        return float(db[i]), i / max(len(db) - 1, 1)

    low_db, low_at = _extremum(4, "min")
    high_db, high_at = _extremum(2, "max")

    # --- rhythm / impact ----------------------------------------------------
    y16 = librosa.resample(mono, orig_sr=sr, target_sr=16000)
    onset = librosa.onset.onset_strength(y=y16, sr=16000)
    onset_env = onset / (onset.max() + 1e-9)
    onset_mean = float(onset_env.mean())
    try:
        tempo = float(librosa.beat.tempo(onset_envelope=onset, sr=16000)[0])
    except Exception:
        tempo = 0.0

    # low-end (40-120 Hz) share of the loudest 4 s vs the body
    def _low_share(seg: np.ndarray) -> float:
        if len(seg) < 4096:
            return 0.0
        mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        tot = mag.sum() + 1e-9
        return float(mag[(freqs >= 40) & (freqs <= 120)].sum() / tot)

    hi_centre = int(high_at * dur * sr)
    body = mono
    low_share_peak = _low_share(mono[max(hi_centre - 2 * sr, 0): hi_centre + 2 * sr])
    low_share_body = _low_share(body)

    out = {
        "file": path.name,
        "duration_s": round(dur, 1),
        "rms_p95_db": round(p95, 1),
        "rms_p50_db": round(p50, 1),
        "rms_p10_db": round(p10, 1),
        "range_db": round(p95 - p10, 1),      # intra-song dynamic spread
        "low_point_db": round(low_db, 1),     # quietest sustained passage
        "low_at": round(low_at, 2),
        "high_point_db": round(high_db, 1),
        "high_at": round(high_at, 2),
        "onset_mean": round(onset_mean, 3),
        "tempo_bpm": round(tempo),
        "low_share_peak": round(low_share_peak, 3),
        "low_share_body": round(low_share_body, 3),
    }

    if with_f0:
        # Melodic pitch range of the lead line: pyin on the 16 kHz mono mix
        # (the vocal dominates the melody, so this tracks the sung line).
        f0, voiced, _ = librosa.pyin(
            y16, fmin=float(librosa.note_to_hz("C3")),
            fmax=float(librosa.note_to_hz("C7")), sr=16000,
            frame_length=2048, hop_length=256)
        v = f0[voiced & np.isfinite(f0)]
        if len(v):
            lo, mid, hi = (float(np.percentile(v, q)) for q in (5, 50, 95))
            out["f0_p5_hz"] = round(lo)
            out["f0_p50_hz"] = round(mid)
            out["f0_p95_hz"] = round(hi)
            out["f0_range_semitones"] = round(12 * np.log2(hi / lo), 1)
            # register contrast: quietest passage vs the climax
            lo_seg = y16[int(low_at * len(y16)): int(low_at * len(y16)) + 4 * 16000]
            f_lo, v_lo, _ = librosa.pyin(lo_seg, fmin=float(librosa.note_to_hz("C3")),
                                         fmax=float(librosa.note_to_hz("C7")),
                                         sr=16000, frame_length=2048, hop_length=256)
            f_lo = f_lo[v_lo & np.isfinite(f_lo)]
            if len(f_lo):
                out["f0_lowpassage_hz"] = round(float(np.median(f_lo)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="*", type=Path)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--no-f0", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    files = list(args.wavs)
    if args.dir:
        files += sorted(p for p in args.dir.glob("*.wav") if "_raw" not in p.name)
    if not files:
        ap.error("no wavs")

    rows = [measure(p, with_f0=not args.no_f0) for p in files]
    hdr = (f"{'file':<30}{'dur':>6}{'rng':>6}{'p50':>7}{'p10':>7}{'low':>7}"
           f"{'low@':>6}{'hi@':>6}{'onset':>7}{'bpm':>6}{'lowpk':>7}"
           f"{'f0lo':>6}{'f0md':>6}{'f0hi':>6}{'semi':>6}")
    print(hdr)
    for r in rows:
        print(f"{r['file'][:29]:<30}{r['duration_s']:>6.1f}{r['range_db']:>6.1f}"
              f"{r['rms_p50_db']:>7.1f}{r['rms_p10_db']:>7.1f}"
              f"{r['low_point_db']:>7.1f}{r['low_at']:>6.2f}{r['high_at']:>6.2f}"
              f"{r['onset_mean']:>7.3f}{r['tempo_bpm']:>6}"
              f"{r['low_share_peak']:>7.3f}"
              f"{r.get('f0_p5_hz', 0):>6}{r.get('f0_p50_hz', 0):>6}"
              f"{r.get('f0_p95_hz', 0):>6}{r.get('f0_range_semitones', 0):>6.1f}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
