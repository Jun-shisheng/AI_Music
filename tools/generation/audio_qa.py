"""Compact objective QA table for a set of WAVs.

These are DIAGNOSTIC proxies, not an acceptance gate. This project's own history
shows spectral proxies cannot tell sibilants from machine noise, so a human must
still pick the take. What these numbers are good for is a fast relative read on:

  clicks   -- sample-level discontinuities (audible as crackle/爆音). Counted on
              both channels separately and where they coincide (a true mastering
              click hits both channels at once; a codec glitch often does too).
  dips     -- >=150 ms near-silent stretches between active audio (断裂/dropouts).
  harsh    -- share of 50 ms frames whose zero-crossing rate is abnormally high,
              i.e. the "machine noise" signature the old repair chain targeted.

Usage:
  python tools/generation/audio_qa.py <wav> [<wav> ...]
  python tools/generation/audio_qa.py --dir outputs/ab_cfg
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import soundfile as sf


def analyze(path: Path) -> dict:
    y, sr = sf.read(path, always_2d=True)
    n, ch = y.shape
    mono = y.mean(axis=1)
    dur = n / sr

    # --- clicks: sample-to-sample jumps above a fixed normalized threshold
    thresh = 0.25
    per_ch = []
    for c in range(ch):
        sig = y[:, c] / max(np.max(np.abs(y[:, c])), 1e-12)
        per_ch.append(np.abs(np.diff(sig)) > thresh)
    clicks = sum(int(m.sum()) for m in per_ch)
    both = int(np.logical_and.reduce(per_ch).sum()) if per_ch else 0

    # --- dropouts: >=150 ms frames below 25% of the 10th-percentile active level
    fl = int(0.05 * sr)
    nf = n // fl
    if nf > 0:
        r = np.sqrt(np.mean(mono[: nf * fl].reshape(nf, fl) ** 2, axis=1))
        act = r[r > r.max() * 0.05]
        thr = (np.percentile(act, 10) * 0.25) if act.size else 0.0
        dips, run = 0, 0
        for v in r:
            if v < thr:
                run += 1
            else:
                if run * 0.05 >= 0.15:
                    dips += 1
                run = 0
        if run * 0.05 >= 0.15:
            dips += 1
    else:
        dips = 0

    # --- harsh frames: high zero-crossing rate in 50 ms frames
    harsh = 0
    if nf > 0:
        seg = mono[: nf * fl].reshape(nf, fl)
        zcr = np.mean(np.abs(np.diff(np.sign(seg), axis=1)) > 0, axis=1)
        harsh = int((zcr > 0.55).sum())

    return {"file": path.name, "dur": dur, "sr": sr, "peak": float(np.max(np.abs(mono))),
            "rms": float(np.sqrt(np.mean(mono ** 2))), "clicks": clicks,
            "clicks_both": both, "dips": dips, "harsh": harsh, "harsh_ratio": harsh / max(nf, 1)}


def tail_degeneracy(path: Path, click_thresh: float = 0.25,
                    tail_fraction: float = 0.2) -> dict:
    """Detect a degenerate ending: a burst of clicks concentrated at the tail.

    When the model runs out of lyrics near the end it sometimes stops tracking
    the music and emits noise, which shows up as a dense cluster of sample-level
    discontinuities in the final section. Measured on the 2026-10-09 batch, a
    healthy take sits at 5-45 clicks/s with <=30% of them in the last fifth; a
    degenerate take reached 326 clicks/s with two thirds of them at the tail.

    Calibrated thresholds: degenerate when >50% of clicks fall in the tail AND
    the overall click rate exceeds 100/s. Meant as a retry trigger, not a
    listening gate.
    """
    y, sr = sf.read(path, always_2d=True)
    mono = y.mean(axis=1)
    m = mono / (float(np.max(np.abs(mono))) or 1.0)
    idx = np.where(np.abs(np.diff(m)) > click_thresh)[0]
    dur = len(m) / sr
    if len(idx) == 0 or dur <= 0:
        return {"clicks": 0, "clicks_per_s": 0.0, "tail_share": 0.0, "degenerate": False}
    tail = int(np.sum(idx / sr >= dur * (1 - tail_fraction)))
    share = tail / len(idx)
    cps = len(idx) / dur
    return {"clicks": len(idx), "clicks_per_s": round(cps, 1),
            "tail_share": round(share, 3), "degenerate": share > 0.5 and cps > 100}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="*", type=Path)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--degeneracy", action="store_true",
                    help="Also report tail-degeneracy (clicks/s, tail share).")
    args = ap.parse_args()

    paths = list(args.wavs)
    if args.dir:
        paths += [Path(p) for p in sorted(glob.glob(str(args.dir / "*.wav")))]
    if not paths:
        ap.error("no wavs given")

    rows = [analyze(p) for p in paths]
    if args.degeneracy:
        for r, p in zip(rows, paths):
            r.update(tail_degeneracy(p))
    head = (f"{'file':<34}{'dur':>7}{'peak':>7}{'rms':>8}{'clicks':>8}{'both':>6}"
            f"{'dips':>6}{'harsh':>7}{'h.ratio':>9}")
    if args.degeneracy:
        head += f"{'clk/s':>8}{'tail':>7}  verdict"
    print(head)
    for r in rows:
        line = (f"{r['file']:<34}{r['dur']:>6.1f}s{r['peak']:>7.3f}{r['rms']:>8.4f}"
                f"{r['clicks']:>8}{r['clicks_both']:>6}{r['dips']:>6}{r['harsh']:>7}"
                f"{r['harsh_ratio']:>9.4f}")
        if args.degeneracy:
            verdict = "DEGENERATE" if r["degenerate"] else "ok"
            line += f"{r['clicks_per_s']:>8.1f}{r['tail_share']*100:>6.1f}%  {verdict}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
