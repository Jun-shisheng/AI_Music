"""Dynamic-arc QA: does a take actually build up before it climaxes?

The user's aesthetic requirement for battle tracks is 欲扬先抑 -- hold back,
then release into an epic climax. Nobody can listen to every candidate, and
this project has no ears in the loop overnight, so this tool turns that
requirement into something measurable:

  intro_db    level of the opening 15 % relative to the loudest section (dB).
              Negative = the track starts restrained. 欲扬先抑 wants this low.
  peak_at     where the loudest 2 s window sits on the timeline (0-1).
              A late peak is what a build-up sounds like structurally.
  lift_db     peak minus intro. Large = a real release, not a flat wall.
  quiet_share share of the track that sits >12 dB below the peak.
              Restraint/breathing room; 0 means wall-to-wall loudness.
  verdict     heuristic label combining the above.

Added 2026-10-15, after the user heard the first English epic batch and called
it flat -- "低的时候不够低，高的时候不够高有点平了" -- while also saying they
keep the rhythmic tracks:

  body_db      median level of the track relative to the peak. On that delivery
               this sat at -2.7 to -6.8 dB, i.e. the song lived just under its
               own maximum: loud almost everywhere, which is exactly what "flat"
               sounds like. Lower is more dynamic.
  rms_spread_db p95-p10 of the 1 s envelope -- the intra-song dynamic spread.
               That delivery measured 11.6-17.5 dB.
  mid_dip_db   quietest sustained (4 s) stretch between 20 % and 90 % of the
               song, relative to the peak. The edges are excluded so an outro
               fade cannot pass as a dramatic drop; this is the real "抑".
  onset_mean   mean normalised spectral attack: how percussive/rhythmic the
               track is. Dependency-free stand-in for librosa's onset strength.
  low_share_peak  share of 40-120 Hz energy in the 4 s around the climax:
               how much sub-bass weight the drop actually carries.
  centroid_*   spectral centroid at the opening vs at the climax (see
               brightness()): does the climax get bigger or merely louder.

The ASCII envelope printed next to each take is a level meter over time --
enough to eyeball the shape of the arc without listening to anything.

Usage:
  python tools/generation/arc_qa.py <wav> [<wav> ...]
  python tools/generation/arc_qa.py --dir outputs/battle_batch
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

SPARK = " .:-=+*#%@"


def envelope(path: Path, win_s: float = 1.0) -> tuple[np.ndarray, float]:
    """Per-window RMS in dB (relative to the loudest window) and the duration."""
    y, sr = sf.read(path, always_2d=True)
    mono = y.mean(axis=1)
    win = max(int(sr * win_s), 1)
    n_win = max(len(mono) // win, 1)
    frames = mono[: n_win * win].reshape(n_win, win)
    rms = np.sqrt(np.maximum((frames ** 2).mean(axis=1), 1e-20))
    db = 20 * np.log10(rms / max(rms.max(), 1e-12))
    return db, len(mono) / sr


def _centroid(seg: np.ndarray, sr: int, max_frames: int = 40) -> float:
    """Mean spectral centroid (Hz) of a passage, sample-capped for speed."""
    win = 4096
    if len(seg) < win * 2:
        return 0.0
    starts = range(0, len(seg) - win, win // 2)
    step = max(len(list(starts)) // max_frames, 1)
    window = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    cents = []
    for i, start in enumerate(range(0, len(seg) - win, win // 2)):
        if i % step:
            continue
        mag = np.abs(np.fft.rfft(seg[start:start + win] * window))
        total = mag.sum()
        if total > 1e-9:
            cents.append(float((freqs * mag).sum() / total))
    return float(np.mean(cents)) if cents else 0.0


def brightness(path: Path, peak_at: float) -> dict:
    """Does the climax get *bigger* or merely louder?

    A real tutti climax adds brass/cymbals/choir (more high-frequency content);
    a climax that is only gain-riding leaves the spectral centre where it was.
    Compare the centroid of the opening with the centroid at the climax.
    """
    y, sr = sf.read(path, always_2d=True)
    mono = y.mean(axis=1)
    n = len(mono)
    intro = mono[: max(int(n * 0.15), 1)]
    centre = int(peak_at * n)
    half = max(int(sr * 2.0), 1)
    peak = mono[max(centre - half, 0): min(centre + half, n)]
    c_intro = _centroid(intro, sr)
    c_peak = _centroid(peak, sr)
    return {
        "centroid_intro_hz": round(c_intro),
        "centroid_peak_hz": round(c_peak),
        "brightness_delta_hz": round(c_peak - c_intro),
    }


def sparkline(db: np.ndarray, width: int = 48) -> str:
    """Compress the envelope to `width` cells and map -36..0 dB onto ASCII."""
    if len(db) > width:
        idx = np.linspace(0, len(db) - 1, width).astype(int)
        db = db[idx]
    t = np.clip((db + 36.0) / 36.0, 0.0, 1.0)
    return "".join(SPARK[int(round(v * (len(SPARK) - 1)))] for v in t)


def _spectral_flux(mono: np.ndarray, sr: int, hop: int = 512,
                   win: int = 2048, target_sr: int = 16000) -> float:
    """Mean positive spectral flux, normalised to its own max (0-1).

    A dependency-free stand-in for librosa's onset strength: how much
    percussive attack the track carries. Added 2026-10-15 with the user's
    comment that they keep the rhythmic tracks.
    """
    if sr != target_sr:                      # cheap decimation is enough here
        step = max(int(round(sr / target_sr)), 1)
        mono = mono[::step]
        sr = sr // step
    if len(mono) < win * 2:
        return 0.0
    window = np.hanning(win)
    prev = None
    flux = []
    for start in range(0, len(mono) - win, hop):
        mag = np.abs(np.fft.rfft(mono[start:start + win] * window))
        if prev is not None:
            flux.append(float(np.maximum(mag - prev, 0.0).sum()))
        prev = mag
    if not flux:
        return 0.0
    flux = np.asarray(flux)
    return float((flux / max(flux.max(), 1e-12)).mean())


def _band_share(seg: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> float:
    """Fraction of spectral magnitude inside one band (sub-bass weight)."""
    if len(seg) < 4096:
        return 0.0
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    total = float(mag.sum()) + 1e-12
    return float(mag[(freqs >= lo_hz) & (freqs <= hi_hz)].sum() / total)


def analyze(path: Path) -> dict:
    db, dur = envelope(path)
    n = len(db)
    if n < 4:
        return {"file": str(path), "error": "too short"}

    intro_len = max(int(n * 0.15), 1)
    peak_win = max(int(round(2.0 / 1.0)), 1)          # 2 s of 1 s windows
    if peak_win >= n:
        peak_win = 1
    # Loudest 2 s stretch, not a single frame: one stray hit should not define
    # the climax.
    kernel = np.ones(peak_win) / peak_win
    smooth = np.convolve(db, kernel, mode="valid")
    peak_idx = int(np.argmax(smooth)) + peak_win // 2

    intro_db = float(db[:intro_len].mean())
    lift_db = float(-intro_db)
    peak_at = float(peak_idx / max(n - 1, 1))
    quiet_share = float((db < -12.0).mean())

    # A classic 欲扬先抑 arc also dips *before* the final climax (the quiet
    # bridge). Measure the deepest suppression in the 15-80 % window relative to
    # the peak, and how far back it sits from the peak.
    lo, hi = max(int(n * 0.15), 1), max(int(n * 0.80), 2)
    pre = db[lo:hi]
    if len(pre) and peak_idx >= lo:
        dip_idx = lo + int(np.argmin(pre))
        dip_db = float(db[dip_idx])
        dip_from_peak = float((peak_idx - dip_idx) / max(n - 1, 1))
    else:
        dip_db, dip_from_peak = 0.0, 0.0

    # --- 2026-10-15: the user's "低的时候不够低，高的时候不够高，有点平了" ----
    # `intro_db`/`lift_db` only compare the opening with the peak. These say how
    # much of the *body* sits below the peak and how deep the mid-song drop is,
    # which is what a verse/bridge that stays loud looks like.
    body_db = float(np.median(db))
    rms_spread_db = float(np.percentile(db, 95) - np.percentile(db, 10))
    # Quietest sustained stretch inside the song (edges excluded, so the outro
    # fade cannot masquerade as a dramatic drop).
    mid_lo, mid_hi = max(int(n * 0.2), 1), max(int(n * 0.9), 2)
    quiet_win = min(4, max(mid_hi - mid_lo, 1))
    if mid_hi - mid_lo > quiet_win:
        mid = db[mid_lo:mid_hi]
        rolled = np.convolve(mid, np.ones(quiet_win) / quiet_win, mode="valid")
        mid_dip_idx = mid_lo + int(np.argmin(rolled))
        mid_dip_db = float(db[mid_dip_idx])
    else:
        mid_dip_db = float(db[mid_lo:mid_hi].min())
    mid_dip_at = float(mid_dip_idx / max(n - 1, 1)) if n > 1 else 0.0

    y, sr = sf.read(path, always_2d=True)
    mono = y.mean(axis=1)
    onset_mean = _spectral_flux(mono, sr)
    centre = int(peak_at * dur * sr)
    half = int(2 * sr)
    low_share_peak = _band_share(mono[max(centre - half, 0):centre + half],
                                 sr, 40.0, 120.0)

    if intro_db <= -8 and peak_at >= 0.5:
        verdict = "build-up"
    elif intro_db <= -6 and peak_at >= 0.4:
        verdict = "mild build"
    elif intro_db >= -4:
        verdict = "flat (no restraint)"
    else:
        verdict = "early peak"

    return {
        "file": str(path),
        "duration_s": round(dur, 1),
        "intro_db": round(intro_db, 1),
        "lift_db": round(lift_db, 1),
        "peak_at": round(peak_at, 2),
        "quiet_share": round(quiet_share, 2),
        "dip_before_peak_db": round(dip_db, 1),
        "dip_from_peak": round(dip_from_peak, 2),
        "body_db": round(body_db, 1),
        "rms_spread_db": round(rms_spread_db, 1),
        "mid_dip_db": round(mid_dip_db, 1),
        "mid_dip_at": round(mid_dip_at, 2),
        "onset_mean": round(onset_mean, 3),
        "low_share_peak": round(low_share_peak, 3),
        "verdict": verdict,
        "envelope": sparkline(db),
        **brightness(path, peak_at),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="*", type=Path)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    files = list(args.wavs)
    if args.dir:
        files += sorted(args.dir.rglob("*.wav"))
    if not files:
        ap.error("no wavs given")

    rows = [analyze(p) for p in files]
    print(f"{'file':<34}{'dur':>6}{'intro':>7}{'lift':>7}{'peak@':>7}"
          f"{'body':>7}{'spread':>7}{'midDip':>7}{'onset':>7}{'lowPK':>7}"
          f"{'d_bright':>9}  verdict")
    for r in rows:
        if "error" in r:
            print(f"{Path(r['file']).name:<34}  {r['error']}")
            continue
        print(f"{Path(r['file']).name[:33]:<34}{r['duration_s']:>6.1f}"
              f"{r['intro_db']:>7.1f}{r['lift_db']:>7.1f}{r['peak_at']:>7.2f}"
              f"{r['body_db']:>7.1f}{r['rms_spread_db']:>7.1f}"
              f"{r['mid_dip_db']:>7.1f}{r['onset_mean']:>7.3f}"
              f"{r['low_share_peak']:>7.3f}"
              f"{r['brightness_delta_hz']:>+9}  {r['verdict']}")
    print()
    for r in rows:
        if "envelope" in r:
            print(f"{Path(r['file']).name[:30]:<31}|{r['envelope']}|")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
