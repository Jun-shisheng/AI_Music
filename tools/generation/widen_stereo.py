"""Optional mid/side widening for YuE2 renders (experimental, not native).

YuE2's native mix measures 0.25-0.37 on `style_qa`'s stereo_width while the
film references (Incarnation, 快哉风) sit at 0.50.  Tags do not move it and
the TSFH NAR LoRA at 0.15 moved it by +0.02, so the only remaining lever is
the mix.  This tool boosts the side signal above `--hp` Hz, leaves mid
(loudness, vocal, bass) untouched, soft-limits the added width and clips only
the rare peak above 0.999.

What it does NOT do: it cannot create instruments that were never panned.
A widened 0.25 track is still a narrow arrangement with more side energy --
judge with ears before replacing anything.

Usage:
  python tools/generation/widen_stereo.py --in a.wav --out b.wav --gain 2.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    ap.add_argument("--gain", type=float, default=2.5,
                    help="side gain above --hp (1.0 = untouched)")
    ap.add_argument("--hp", type=float, default=300.0,
                    help="Hz; below this the side stays as-is (keeps bass mono)")
    ap.add_argument("--soft", type=float, default=0.5,
                    help="tanh knee for the side signal")
    args = ap.parse_args()

    y, sr = sf.read(args.src, always_2d=True)
    mid = (y[:, 0] + y[:, 1]) / 2
    side = (y[:, 0] - y[:, 1]) / 2
    sos = butter(2, args.hp / (sr / 2), btype="highpass", output="sos")
    side_hi = sosfilt(sos, side)
    side2 = side_hi * args.gain + (side - side_hi)
    side2 = np.tanh(side2 / args.soft) * args.soft
    left = np.clip(mid + side2, -0.999, 0.999)
    right = np.clip(mid - side2, -0.999, 0.999)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.dst, np.stack([left, right], axis=1).astype(np.float32),
             sr, subtype="PCM_24")
    corr = float(np.corrcoef(left, right)[0, 1])
    clipped = int((np.abs(mid + side2) > 0.999).sum())
    print(f"{args.dst.name}: gain {args.gain} hp {args.hp} corr {corr:.3f} "
          f"clipped {clipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
