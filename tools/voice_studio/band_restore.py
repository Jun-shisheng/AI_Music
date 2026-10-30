"""Rebuild a 24 kHz SVC vocal's high band from a full-band donor stem.

SoulX-SVC runs at 24 kHz, so its output has no content above 12 kHz. The
delivered fixed-voice mixes previously only upsample that vocal, which leaves a
hard 12 kHz cliff and is part of what the user hears as a mechanical high note.
The full-band (44.1/48 kHz) Demucs vocal of the *same* target take is available
and time-aligned, so the top octave can be taken from it without touching the
converted low band.

Crossover is complementary and zero-phase:

    out = lowpass(svc) + (donor - lowpass(donor))

Below the cutoff the result is the SVC voice; above it the result is the donor.
`filtfilt` makes the FIR zero-phase, so the two bands sum without a phase notch.
This is bandwidth restoration, not an EQ or a repair chain; the input files are
never modified and the caller decides whether the result is deliverable.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import filtfilt, firwin, resample_poly

DEFAULT_CUTOFF_HZ = 11000.0
DEFAULT_TAPS = 2049
MAX_LENGTH_MISMATCH_SECONDS = 0.05


def _mono(data: np.ndarray) -> np.ndarray:
    value = np.asarray(data, dtype=np.float64)
    if value.ndim == 2:
        value = value.mean(axis=1)
    if value.ndim != 1:
        raise ValueError("Expected mono or (frames, channels) audio")
    if not np.isfinite(value).all():
        raise ValueError("Audio contains NaN or infinite samples")
    return value


def resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample with an exact rational ratio and preserve the timeline length."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if source_rate == target_rate:
        return np.asarray(data, dtype=np.float64).copy()
    common = gcd(int(source_rate), int(target_rate))
    up, down = int(target_rate) // common, int(source_rate) // common
    value = np.asarray(data, dtype=np.float64)
    if value.ndim == 1:
        return resample_poly(value, up, down)
    return resample_poly(value, up, down, axis=0)


def hf_share(data: np.ndarray, rate: int, low_hz: float, high_hz: float) -> float:
    """Share of total spectral energy in [low_hz, high_hz).

    A Hann window avoids the spectral leakage that would make a 12 kHz cliff
    look softer than it is.
    """
    mono = _mono(data)
    if len(mono) < 2:
        return 0.0
    window = np.hanning(len(mono))
    spectrum = np.abs(np.fft.rfft(mono * window)) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1.0 / rate)
    total = float(spectrum.sum())
    if total <= 0:
        return 0.0
    band = (freqs >= low_hz) & (freqs < high_hz)
    return float(spectrum[band].sum() / total)


def restore_high_band(
    svc: np.ndarray,
    svc_rate: int,
    donor: np.ndarray,
    donor_rate: int,
    *,
    out_rate: int | None = None,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    taps: int = DEFAULT_TAPS,
) -> tuple[np.ndarray, dict]:
    """Return a full-band vocal and a manifest-ready report.

    `svc` is the 24 kHz converted vocal (mono or stereo); `donor` is the
    full-band vocal stem of the same take. Both must cover the same timeline.
    """
    target_rate = int(out_rate or donor_rate)
    if target_rate < 2 * cutoff_hz:
        raise ValueError("Output rate must be above twice the cutoff")

    svc_out = resample(_mono(svc), svc_rate, target_rate)
    donor_out = resample(_mono(donor), donor_rate, target_rate)

    tolerance = int(MAX_LENGTH_MISMATCH_SECONDS * target_rate)
    difference = len(svc_out) - len(donor_out)
    if abs(difference) > tolerance:
        raise ValueError(
            f"SVC and donor timelines differ by {difference} samples at {target_rate} Hz "
            f"(limit {tolerance})"
        )
    if difference > 0:
        svc_out = svc_out[: len(donor_out)]
    elif difference < 0:
        svc_out = np.pad(svc_out, (0, -difference))

    taps = int(taps)
    if taps % 2 == 0:
        taps += 1
    if taps < 31:
        raise ValueError("At least 31 FIR taps are required")
    kernel = firwin(taps, cutoff_hz, fs=target_rate, window="blackman")
    low_svc = filtfilt(kernel, 1.0, svc_out)
    low_donor = filtfilt(kernel, 1.0, donor_out)
    # The converted voice can sit at a different level than the donor. Without
    # this constant low-band gain the donor's high band would be correct in
    # absolute terms but too quiet relative to the SVC body, and the delivered
    # vocal would still sound dull. This is one gain for the whole clip.
    svc_rms = float(np.sqrt(np.mean(low_svc ** 2))) if len(low_svc) else 0.0
    donor_rms = float(np.sqrt(np.mean(low_donor ** 2))) if len(low_donor) else 0.0
    low_band_gain = (donor_rms / svc_rms) if svc_rms > 1e-9 and donor_rms > 1e-9 else 1.0
    restored = low_svc * low_band_gain + (donor_out - low_donor)

    if not np.isfinite(restored).all():
        raise ValueError("Restoration produced non-finite samples")
    peak = float(np.max(np.abs(restored))) if len(restored) else 0.0
    if peak > 1.0:
        restored = restored / peak

    report = {
        "method": "complementary_zero_phase_fir",
        "svc_rate": int(svc_rate),
        "donor_rate": int(donor_rate),
        "out_rate": target_rate,
        "cutoff_hz": float(cutoff_hz),
        "taps": taps,
        "length": int(len(restored)),
        "length_adjust_samples": int(abs(difference)),
        "peak_before_limit": peak,
        "low_band_gain": round(float(low_band_gain), 6),
        "low_band_gain_db": round(float(20.0 * np.log10(low_band_gain)), 4) if low_band_gain > 0 else None,
        "hf_share_11k_before": round(hf_share(svc_out, target_rate, 11000, target_rate / 2), 6),
        "hf_share_11k_after": round(hf_share(restored, target_rate, 11000, target_rate / 2), 6),
        "hf_share_12k_before": round(hf_share(svc_out, target_rate, 12000, target_rate / 2), 6),
        "hf_share_12k_after": round(hf_share(restored, target_rate, 12000, target_rate / 2), 6),
        "time_varying_processing": False,
        "note": "Above the cutoff the samples come from the donor stem, not from the converted voice.",
    }
    return restored.astype(np.float64), report
