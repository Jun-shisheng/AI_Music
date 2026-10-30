"""Diagnose pitch-transition ("转音台阶") shapes on a vocal stem.

Extracts F0 with pyworld on the demucs-separated vocal stem, finds F0 jumps
> 2 semitones between stable notes, and prints the contour shape around each
jump so the user can classify the defect by ear:

    python transition_diag.py <song.wav> [--jump-st 2.0] [--window-s 2.0]

Outputs:
- A table of detected transitions (time, direction, magnitude, duration).
- An ASCII contour of F0 (semitones vs time) around each transition.
- Extracted 4s listening clips under outputs/transition_listen/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "vocal_repair"))

from stem_pipeline import separate_with_demucs  # noqa: E402

JUMP_ST = 2.0
MIN_NOTE_S = 0.15     # a stable note must hold at least this long
GLIDE_MAX_MS = 300.0  # a transition must complete within this window
F0_MIN = 90.0
F0_MAX = 900.0
HOP = 256
N_FFT = 2048
LISTEN_WINDOW_S = 4.0


def extract_f0(x: np.ndarray, sr: int) -> np.ndarray:
    import pyworld as pw
    frame_period = HOP * 1000.0 / sr
    f0_raw, _ = pw.harvest(x.astype(np.float64), sr, frame_period=frame_period,
                           f0_floor=F0_MIN, f0_ceil=F0_MAX)
    return f0_raw


def find_transitions(f0: np.ndarray, sr: int) -> list[dict]:
    """Find note-to-note F0 jumps > JUMP_ST where both sides are stable."""
    frame_s = HOP / sr
    st = lambda hz: 12 * np.log2(max(hz, 1.0) / 440.0) + 69.0  # noqa: E731
    n = len(f0)
    voiced = f0 > 0

    trans = []
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        # start of a voiced run
        run_end = i
        while run_end < n and voiced[run_end]:
            run_end += 1
        run = f0[i:run_end]
        # split the run into segments where F0 stays within ±1 st of the
        # running median (stable notes)
        segs = []  # (start_idx, end_idx, median_f0)
        seg_start = i
        med = run[0]
        for k in range(1, len(run)):
            if abs(st(run[k]) - st(med)) > 1.0:
                segs.append((seg_start, i + k, med))
                seg_start = i + k
                med = run[k]
            else:
                med = np.median(run[: k + 1])
        segs.append((seg_start, run_end, med))
        for a, b in zip(segs[:-1], segs[1:]):
            dur_a = (b[0] - a[0]) * frame_s
            dur_b = (b[1] - b[0]) * frame_s
            if dur_a < MIN_NOTE_S or dur_b < MIN_NOTE_S:
                continue
            gap_s = (b[0] - a[1]) * frame_s
            if gap_s * 1000.0 > GLIDE_MAX_MS:
                continue
            jump = st(b[2]) - st(a[2])
            if abs(jump) < JUMP_ST:
                continue
            # peak excursion during the transition window (between the notes)
            seg = f0[a[1]:b[0]]
            if len(seg) == 0:
                continue
            excursion = max(seg) - min(seg) if len(seg) > 0 else 0.0
            trans.append({
                "t_start": a[1] * frame_s,
                "t_end": b[0] * frame_s,
                "from_hz": float(a[2]),
                "to_hz": float(b[2]),
                "jump_st": float(jump),
                "gap_ms": gap_s * 1000.0,
                "excursion_st": float(excursion),
            })
        i = run_end
    return trans


def ascii_contour(f0: np.ndarray, sr: int, t0: float, t1: float) -> str:
    """Print an ASCII F0 contour (semitones) over [t0, t1]."""
    frame_s = HOP / sr
    i0, i1 = max(0, int(t0 / frame_s)), min(len(f0), int(t1 / frame_s) + 1)
    if i1 <= i0:
        return "(empty)"
    seg = f0[i0:i1]
    st = np.array([12 * np.log2(max(h, 1.0) / 440.0) + 69.0 if h > 0 else np.nan
                   for h in seg])
    vmin, vmax = np.nanmin(st), np.nanmax(st)
    if not np.isfinite(vmin) or vmax - vmin < 0.5:
        return "(no voiced pitch in window)"
    rows = 14
    lines = []
    for r in range(rows - 1, -1, -1):
        thresh = vmin + (vmax - vmin) * r / (rows - 1)
        line = "|"
        for v in st:
            if np.isnan(v):
                line += " "
            elif v >= thresh:
                line += "#"
            else:
                line += " "
        lines.append(f"{vmin + (vmax - vmin) * r / (rows - 1):6.1f} {line}")
    lines.append(f"        +{'-' * (i1 - i0)}")
    lines.append(f"        t={t0:.2f}s{' ' * (i1 - i0 - 9)}t={t1:.2f}s")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("song", type=Path)
    parser.add_argument("--jump-st", type=float, default=JUMP_ST)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--listen-dir", type=Path, default=PROJECT_ROOT / "outputs" / "transition_listen")
    args = parser.parse_args()

    work = PROJECT_ROOT / "outputs" / "transition_work" / args.song.stem
    vocals, _ = separate_with_demucs(args.song, work)
    y, sr = sf.read(vocals, always_2d=True)
    x = y.mean(axis=1).astype(np.float64) if y.ndim > 1 else y.astype(np.float64)

    print(f"F0 extraction on vocal stem ({sr} Hz, {len(x) / sr:.1f} s)...")
    f0 = extract_f0(x, sr)
    trans = find_transitions(f0, sr)
    print(f"transitions > {args.jump_st} st: {len(trans)}\n")

    listen_dir = args.listen_dir
    listen_dir.mkdir(parents=True, exist_ok=True)
    for k, t in enumerate(trans, 1):
        mid = (t["t_start"] + t["t_end"]) / 2.0
        w = args.window_s
        t0, t1 = mid - w, mid + w
        i0, i1 = max(0, int(t0 * sr)), min(len(x), int(t1 * sr))
        clip = work / f"t{k:02d}_at{mid:.1f}s.wav"
        sf.write(clip, x[i0:i1], sr, subtype="PCM_16")
        print(f"[{k:02d}] t={t['t_start']:.2f}->{t['t_end']:.2f}s  "
              f"{t['from_hz']:.0f}->{t['to_hz']:.0f} Hz  "
              f"{t['jump_st']:+.1f} st  gap={t['gap_ms']:.0f} ms  clip={clip.name}")
        print(ascii_contour(f0, sr, t0, t1))
        print()

    if not args.listen_dir and work.exists():
        import shutil
        shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
