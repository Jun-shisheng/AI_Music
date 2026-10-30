"""Objective style profile for the epic / ethereal line (diagnostic + ranking).

The 2026-10-26 Chinese batch was rejected by ear as 糖水歌.  Measuring it next
to the twelve local epic references (human_refs/refs/) shows what "sugar
water" is numerically -- not major keys or slow tempos (the batch was mostly
minor already), but a soft, dark, narrow mix:

  metric                  our 8 takes      epic references
  stereo width            0.25-0.56        0.38-0.84
  2-6 kHz minus 0.5-2k    -11 .. -17 dB    -7 .. -13 dB
  sub 30-120 Hz           +27 .. +30 dB    +19 .. +38 dB (most >= 29)

So this tool reports key/mode, tempo, dynamics, width, band balance, and a
0-100 "epic profile" score calibrated on those references.  It is a candidate
ranker and a red flag for a mix that is too dark/narrow -- never the final
verdict (the human still listens).

Usage:
  python -I tools/generation/style_qa.py --dir outputs/2026-10-28_epic/selected
  python -I tools/generation/style_qa.py --file song.wav --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _key_of(y, sr):
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    prof = chroma.mean(axis=1)
    prof = (prof - prof.mean()) / (prof.std() + 1e-9)
    best = (-9.0, None)
    for i in range(12):
        for mode, tmpl in (("major", MAJOR), ("minor", MINOR)):
            t = np.roll((tmpl - tmpl.mean()) / tmpl.std(), i)
            r = float(np.dot(prof, t) / 12)
            if r > best[0]:
                best = (r, f"{NAMES[i]} {mode}")
    return best[1], round(best[0], 3)


def _band_db(S, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    if not m.any():
        return -120.0
    return float(10 * np.log10(np.mean(S[m] ** 2) + 1e-12))


def _ramp(value: float, low: float, high: float) -> float:
    """Map low->0, high->1, clamped."""
    if high == low:
        return 1.0 if value >= high else 0.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


def probe(path: str | Path) -> dict:
    import librosa

    path = Path(path)
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    dur = len(y) / sr
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    per_s = np.array([rms[i:i + 43].mean() for i in range(0, len(rms) - 43, 43)])
    per_s = np.maximum(per_s, 1e-6)
    db = 20 * np.log10(per_s)
    dyn = float(np.percentile(db, 95) - np.percentile(db, 10))
    quiet_share = float(np.mean(db < np.percentile(db, 95) - 12))

    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0]).ravel()[0])
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))

    y2, _ = librosa.load(str(path), sr=22050, mono=False)
    if getattr(y2, "ndim", 1) == 2:
        mid = (y2[0] + y2[1]) / 2
        side = (y2[0] - y2[1]) / 2
        width = float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-9))
    else:
        width = 0.0

    key, conf = _key_of(y, sr)
    # Krumhansl-style profiles confuse relative major/minor on pop mixes (the
    # 02 take was planned in E minor but reads as A major).  Report both so a
    # "major" reading is not mistaken for a major-key song.
    relative = None
    if key and key.endswith("major"):
        tonic = NAMES.index(key.split()[0])
        relative = f"{NAMES[(tonic - 3) % 12]} minor"
    bands = {
        "sub": _band_db(S, freqs, 30, 120),
        "lowmid": _band_db(S, freqs, 120, 500),
        "mid": _band_db(S, freqs, 500, 2000),
        "himid": _band_db(S, freqs, 2000, 6000),
        "high": _band_db(S, freqs, 6000, 12000),
    }
    presence = bands["himid"] - bands["mid"]

    # 0-100 epic profile, calibrated on human_refs/refs/ (12 tracks).
    score = 100.0 * (
        0.35 * _ramp(width, 0.32, 0.68)          # width
        + 0.35 * _ramp(presence, -17.0, -8.5)    # 2-6k vs 0.5-2k
        + 0.15 * _ramp(bands["sub"], 24.0, 34.0)  # sub weight
        + 0.15 * _ramp(centroid, 1500.0, 2400.0)  # orchestral brightness
    )
    # The label is deliberately weak: 12 accepted English epic YuE2 takes span
    # 29-60 on this scale, so the score flags a very narrow/dark mix, it does
    # not certify "epic".  The human ear still decides.
    verdict = "wide" if score >= 60 else "typical" if score >= 35 else "narrow"
    return {
        "file": str(path),
        "duration_s": round(dur, 1),
        "key": key,
        "key_conf": conf,
        "relative_minor": relative,
        "minor": key is not None and key.endswith("minor"),
        "tempo_bpm": round(tempo, 1),
        "dyn_range_db": round(dyn, 1),
        "quiet_share": round(quiet_share, 3),
        "centroid_hz": round(centroid),
        "stereo_width": round(width, 3),
        **{f"{k}_db": round(v, 1) for k, v in bands.items()},
        "presence_db": round(presence, 1),
        "epic_score": round(score, 1),
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    paths = ([args.file] if args.file
             else sorted(args.dir.glob("*.wav")) if args.dir else [])
    if not paths:
        print("give --file or --dir", file=sys.stderr)
        return 2
    rows = [probe(p) for p in paths]

    hdr = ["file", "key", "tempo_bpm", "dyn_range_db", "stereo_width",
           "presence_db", "sub_db", "centroid_hz", "epic_score", "verdict"]
    print(" | ".join(h[:14] for h in hdr))
    for r in rows:
        print(" | ".join(str(r.get(h))[:14] for h in hdr))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
