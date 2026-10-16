"""Single source of truth for loudness normalization.

Before this module existed the same 8 lines were copy-pasted into five
generation entry points (clean_pipeline, batch_tuned, ab_cfg,
batch_generate_20_resident, generate_selected_full), and every copy carried the
same bug: `gain = min(target / rms, 4.0)`.

Why that bug mattered (measured 2026-10-10 on the 20-song batch): HeartMuLa's
native output is quiet (raw RMS ~0.03 on the mono blend), so reaching the
-16 dBFS target needed MORE than 4x. The cap silently clipped the gain, and 17 of
the 20 delivered songs ended up below target -- the worst by 2.6 dB. Within one
batch that is a systematic penalty on whichever songs happen to be quietest, and
loudness is one of the heaviest single factors in how good a mix is judged to be.

Proof that the boost cap (not the peak ceiling) was the cause: had the peak
ceiling bound, the delivered peak would sit at exactly 0.99. Songs 19/16/11/02/01
landed at 0.848/0.776/0.881/0.856/0.921, so the ceiling could not have fired.

The fix is threefold: one implementation, a higher boost cap (+18 dB instead of
+12 dB), and a RETURNED REPORT so `rms_in` / the applied gain land in the manifest.
The old code recorded neither, which is exactly why the bug could not be proven
from the artifacts after the raw files were deleted.

Normalization stays a single linear gain: it does not touch the spectrum, which
is the pipeline's zero-post-processing rule (see clean_pipeline's docstring).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_DBFS = -16.0
MAX_BOOST_DB = 18.0  # +18 dB; the old copies capped at +12 dB (4.0x)
PEAK_CEILING = 0.99


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-12)))


def measure(path: Path) -> dict:
    """Level of a file: RMS of the mono blend plus the true peak."""
    y, _sr = sf.read(path, always_2d=True)
    mono = y.mean(axis=1)
    return {
        "rms": float(np.sqrt(np.mean(mono**2))),
        "peak": float(np.max(np.abs(y))) if y.size else 0.0,
        "channels": int(y.shape[1]),
    }


def normalize_loudness(
    src: Path,
    dst: Path,
    *,
    target_dbfs: float = TARGET_DBFS,
    max_boost_db: float = MAX_BOOST_DB,
    peak_ceiling: float = PEAK_CEILING,
) -> dict:
    """Linear-gain loudness normalization. Returns the report for the manifest.

    Measures on the mono blend (so a wide stereo mix is not judged louder than a
    narrow one) and applies one gain to every channel, which preserves the stereo
    image exactly. Two ceilings can bind, and the report says which did:

      boost_capped  -- the target would need more than `max_boost_db` of gain
      peak_limited  -- reaching the target would push the peak over `peak_ceiling`
    """
    y, sr = sf.read(src, always_2d=True)
    if y.size == 0:
        raise ValueError(f"{src} is empty")

    mono = y.mean(axis=1)
    rms_in = float(np.sqrt(np.mean(mono**2)))
    peak_in = float(np.max(np.abs(y)))

    max_gain = 10 ** (max_boost_db / 20)
    gain = (10 ** (target_dbfs / 20)) / (rms_in + 1e-9)
    boost_capped = gain > max_gain
    gain = min(gain, max_gain)

    peak_limited = peak_in * gain > peak_ceiling
    if peak_limited:
        gain = peak_ceiling / (peak_in + 1e-12)

    out = y * gain
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, out, sr, subtype="PCM_16")

    rms_out = float(np.sqrt(np.mean(out.mean(axis=1) ** 2)))
    return {
        "rms_in": rms_in,
        "rms_in_dbfs": _db(rms_in),
        "rms_out": rms_out,
        "rms_out_dbfs": _db(rms_out),
        "target_dbfs": target_dbfs,
        "gain_linear": float(gain),
        "gain_db": _db(gain),
        "peak_in": peak_in,
        "peak_out": float(np.max(np.abs(out))),
        "boost_capped": bool(boost_capped),
        "peak_limited": bool(peak_limited),
        "shortfall_db": float(target_dbfs - _db(rms_out)),
    }


def match_to_level(src: Path, dst: Path, target_rms: float,
                   peak_ceiling: float = PEAK_CEILING) -> dict:
    """Two-point loudness match: bring `src` to an already-measured `target_rms`.

    Used for A/B packs. Normalizing each arm to a fixed dBFS target is NOT the
    same thing: a take whose peak ceiling binds ends up quieter than its partner,
    and in a listening test the louder arm wins regardless of the music. Matching
    one arm to the other's measured level removes that.
    """
    y, sr = sf.read(src, always_2d=True)
    if y.size == 0:
        raise ValueError(f"{src} is empty")
    mono = y.mean(axis=1)
    rms_in = float(np.sqrt(np.mean(mono**2)))
    peak_in = float(np.max(np.abs(y)))

    gain = target_rms / (rms_in + 1e-9)
    peak_limited = peak_in * gain > peak_ceiling
    if peak_limited:
        gain = peak_ceiling / (peak_in + 1e-12)

    out = y * gain
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, out, sr, subtype="PCM_16")
    return {
        "rms_in": rms_in, "rms_out": float(np.sqrt(np.mean(out.mean(axis=1) ** 2))),
        "gain_linear": float(gain), "gain_db": _db(gain),
        "peak_in": peak_in, "peak_out": float(np.max(np.abs(out))),
        "peak_limited": bool(peak_limited),
    }
