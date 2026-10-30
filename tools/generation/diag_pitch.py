"""Compare the sung register of takes over the same slice of the timeline.

Written 2026-10-15 to answer one question objectively: when the plan's final
chorus is transposed up (`yue2_plan_edit.transpose_section`), does the rendered
vocal actually come back higher? `arc_qa.brightness` measures spectral centroid,
which moves with the arrangement as much as with the voice, so it cannot settle
this on its own.

pyin runs on a 16 kHz mono fold of the mix; the lead vocal dominates the melody,
so the estimate tracks the sung line (octave errors are possible, which is why
two takes are always compared over the *same* window rather than in absolute
terms).

Usage:
  python tools/generation/diag_pitch.py --window 0.78 0.92 a.wav b.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def window_f0(path: Path, lo: float, hi: float) -> dict:
    import librosa

    y, sr = sf.read(path, always_2d=True)
    mono = y.mean(axis=1).astype(np.float32)
    dur = len(mono) / sr
    seg = mono[int(lo * dur * sr):int(hi * dur * sr)]
    if len(seg) < sr:
        return {"file": path.name, "error": "window too short", "duration_s": round(dur, 1)}
    y16 = librosa.resample(seg, orig_sr=sr, target_sr=16000)
    f0, voiced, _ = librosa.pyin(
        y16, fmin=float(librosa.note_to_hz("C3")),
        fmax=float(librosa.note_to_hz("C7")), sr=16000,
        frame_length=2048, hop_length=256)
    v = f0[voiced & np.isfinite(f0)]
    if not len(v):
        return {"file": path.name, "error": "no voiced frames", "duration_s": round(dur, 1)}
    return {
        "file": path.name,
        "duration_s": round(dur, 1),
        "window_s": [round(lo * dur, 1), round(hi * dur, 1)],
        "median_hz": round(float(np.median(v))),
        "p90_hz": round(float(np.percentile(v, 90))),
        "p99_hz": round(float(np.percentile(v, 99))),
        "voiced_share": round(float(len(v) / max(len(f0), 1)), 2),
    }


def semitones(a: float, b: float) -> float:
    return round(12 * np.log2(b / max(a, 1e-9)), 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+", type=Path)
    ap.add_argument("--window", nargs=2, type=float, default=[0.78, 0.92],
                    metavar=("LO", "HI"))
    args = ap.parse_args()

    rows = [window_f0(p, *args.window) for p in args.wavs]
    print(f"window {args.window[0]:.2f}-{args.window[1]:.2f} of each file's own duration")
    print(f"{'file':<34}{'median':>8}{'p90':>8}{'p99':>8}{'voiced':>8}")
    for r in rows:
        if "error" in r:
            print(f"{r['file'][:33]:<34}  {r['error']}")
            continue
        print(f"{r['file'][:33]:<34}{r['median_hz']:>8}{r['p90_hz']:>8}"
              f"{r['p99_hz']:>8}{r['voiced_share']:>8}")
    if len(rows) == 2 and all("median_hz" in r for r in rows):
        delta = semitones(rows[0]["median_hz"], rows[1]["median_hz"])
        print(f"\n{rows[1]['file']} vs {rows[0]['file']}: {delta:+.2f} semitones "
              f"on the median")
    return 0


if __name__ == "__main__":
    sys.exit(main())
