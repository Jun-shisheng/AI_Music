"""Measurable evidence for the high-note / melisma complaints.

The user described the problem as "high notes and runs turn mechanical, not
like a human crack". The old QA could not see it: `audio.technical_qa` only
checks duration, dropouts and clipping, and the legacy ACE/HeartMuLa detectors
(zero-crossing rate) are blind on YuE2. `docs/arrangement_spec.md` §2 records
the two working hypotheses:

* H1 - at high F0 the harmonic series gets sparse and the decoder fills the
  gaps between harmonics with inharmonic energy (8-16 kHz);
* H2 - long high notes are rendered as a stable, vibrato-free tone.

This module measures those, plus the 24 kHz bandwidth cliff that the SoulX-SVC
path introduced. It reports evidence and flags; it never claims a take sounds
good and it never substitutes for listening. Thresholds are `warn` until they
have been calibrated against the user's own defect timestamps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

F0_SAMPLE_RATE = 24000
F0_FRAME = 2048
F0_HOP = 256
HIGH_NOTE_HZ = 800.0
MIN_HIGH_SEGMENT_SECONDS = 0.25
MIN_VIBRATO_SEGMENT_SECONDS = 0.35
VIBRATO_BAND_HZ = (4.0, 7.0)
FLAT_VIBRATO_CENTS = 15.0


def _mono(data: np.ndarray) -> np.ndarray:
    value = np.asarray(data, dtype=np.float64)
    if value.ndim == 2:
        value = value.mean(axis=1)
    if value.ndim != 1:
        raise ValueError("Expected mono or (frames, channels) audio")
    if not np.isfinite(value).all():
        raise ValueError("Audio contains NaN or infinite samples")
    return value


def _load(path: Path | str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    data, rate = sf.read(str(path), dtype="float64", always_2d=False)
    return _mono(data), int(rate)


def _resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    from math import gcd
    from scipy.signal import resample_poly
    if source_rate == target_rate:
        return data.astype(np.float64, copy=False)
    common = gcd(int(source_rate), int(target_rate))
    return resample_poly(data.astype(np.float64), target_rate // common, source_rate // common)


def hf_share(data: np.ndarray, rate: int, low_hz: float, high_hz: float) -> float:
    mono = _mono(data)
    if len(mono) < 2 or high_hz <= low_hz:
        return 0.0
    window = np.hanning(len(mono))
    power = np.abs(np.fft.rfft(mono * window)) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1.0 / rate)
    total = float(power.sum())
    if total <= 0:
        return 0.0
    band = (freqs >= low_hz) & (freqs < high_hz)
    return float(power[band].sum() / total)


def _semitone(hz: np.ndarray) -> np.ndarray:
    return 12.0 * np.log2(np.maximum(hz, 1.0) / 440.0)


def _contour(data: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pyin at 24 kHz; f0, voiced probability and frame times."""
    import librosa
    y = _resample(data, rate, F0_SAMPLE_RATE)
    f0, voiced, probability = librosa.pyin(
        y, fmin=65.0, fmax=1600.0, sr=F0_SAMPLE_RATE,
        frame_length=F0_FRAME, hop_length=F0_HOP)
    times = librosa.times_like(f0, sr=F0_SAMPLE_RATE, hop_length=F0_HOP)
    f0 = np.where(np.asarray(voiced, dtype=bool), f0, np.nan)
    return f0.astype(np.float64), np.asarray(voiced, dtype=bool), times


def _segments(mask: np.ndarray, times: np.ndarray, min_seconds: float,
              max_gap_seconds: float = 0.05) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and times[start] - times[merged[-1][1] - 1] <= max_gap_seconds:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    result = []
    for start, end in merged:
        if end - start < 2:
            continue
        duration = times[end - 1] - times[start]
        if duration >= min_seconds:
            result.append((start, end))
    return result


def _vibrato(f0_slice: np.ndarray, frame_rate: float) -> tuple[float | None, float | None]:
    """Vibrato depth in cents and rate in Hz for one sustained note."""
    if len(f0_slice) < max(8, int(0.35 * frame_rate)):
        return None, None
    index = np.arange(len(f0_slice))
    voiced = np.isfinite(f0_slice)
    if voiced.sum() < 8:
        return None, None
    filled = np.interp(index, index[voiced], f0_slice[voiced])
    cents = 1200.0 * np.log2(np.maximum(filled, 1.0) / 440.0)
    # Remove the note's own contour (about 1 s) so only modulation remains.
    # A shorter boxcar would also attenuate the 4-7 Hz vibrato itself.
    width = max(3, int(1.0 * frame_rate))
    if len(cents) >= width:
        kernel = np.ones(width) / width
        smooth = np.convolve(cents, kernel, mode="same")
    else:
        smooth = np.full_like(cents, float(np.mean(cents)))
    residual = cents - smooth
    from scipy.signal import butter, sosfiltfilt
    if frame_rate <= 2 * VIBRATO_BAND_HZ[1]:
        return None, None
    sos = butter(4, VIBRATO_BAND_HZ, btype="bandpass", fs=frame_rate, output="sos")
    band = sosfiltfilt(sos, residual)
    depth = float(np.sqrt(2.0) * np.sqrt(np.mean(band ** 2)))
    spectrum = np.abs(np.fft.rfft(band * np.hanning(len(band))))
    freqs = np.fft.rfftfreq(len(band), 1.0 / frame_rate)
    search = (freqs >= 3.0) & (freqs <= 9.0)
    rate_hz = float(freqs[search][int(np.argmax(spectrum[search]))]) if search.any() else None
    return depth, rate_hz


def _hnr(data: np.ndarray, rate: int) -> float | None:
    """Median harmonics-to-noise ratio in one segment (autocorrelation)."""
    if len(data) < rate // 10:
        return None
    frame = max(256, int(0.04 * rate))
    hop = max(64, int(0.01 * rate))
    values = []
    for start in range(0, len(data) - frame + 1, hop):
        x = data[start:start + frame]
        x = x - x.mean()
        energy = float(np.dot(x, x))
        if energy <= 1e-12:
            continue
        correlation = np.correlate(x, x, mode="full")[frame - 1:]
        correlation = correlation / (correlation[0] + 1e-12)
        low = max(1, int(rate / 1600.0))
        high = min(len(correlation) - 1, int(rate / 65.0))
        if high <= low:
            continue
        best = float(np.max(correlation[low:high]))
        best = float(np.clip(best, 1e-6, 1.0 - 1e-6))
        values.append(10.0 * np.log10(best / (1.0 - best)))
    return float(np.median(values)) if values else None


def _harmonic_ratio(data: np.ndarray, rate: int, f0_slice: np.ndarray, frame_rate: float) -> float | None:
    """Energy at harmonic bins vs between them, in dB (1-8 kHz)."""
    if len(data) < 512 or np.isfinite(f0_slice).sum() < 3:
        return None
    n_fft = 1024
    hop = 256
    if len(data) < n_fft:
        return None
    frames = 1 + (len(data) - n_fft) // hop
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / rate)
    ratios = []
    for index in range(frames):
        center = min(len(f0_slice) - 1, int(index * hop / (rate / frame_rate)))
        f0 = f0_slice[center]
        if not np.isfinite(f0) or f0 < 80.0:
            continue
        spectrum = np.abs(np.fft.rfft(data[index * hop:index * hop + n_fft] * window)) ** 2
        band = (freqs >= 200.0) & (freqs <= min(8000.0, rate / 2))
        if not band.any():
            continue
        harmonic = np.zeros_like(band)
        count = int(min(8000.0, rate / 2) // f0)
        for k in range(max(1, int(200 // f0)), count + 1):
            bin_index = int(round(k * f0 * n_fft / rate))
            if bin_index <= 0 or bin_index >= len(freqs):
                continue
            # +/-2 bins covers the Hann main lobe; scaling the width with F0
            # would overlap neighbouring harmonics and leave no inter-harmonic
            # bins to compare against.
            width = 2
            harmonic[max(0, bin_index - width):bin_index + width + 1] = True
        harmonic &= band
        inter = band & ~harmonic
        if not harmonic.any() or not inter.any():
            continue
        h = float(spectrum[harmonic].sum())
        i = float(spectrum[inter].sum())
        if h > 0 and i > 0:
            ratios.append(10.0 * np.log10(h / i))
    return float(np.median(ratios)) if ratios else None


def _flatness(data: np.ndarray, rate: int) -> float | None:
    if len(data) < 512:
        return None
    import librosa
    values = librosa.feature.spectral_flatness(y=data.astype(np.float32), n_fft=1024, hop_length=256)[0]
    return float(np.median(values)) if len(values) else None


def _cliff_db(data: np.ndarray, rate: int) -> float | None:
    low = hf_share(data, rate, 11000, 12000)
    high = hf_share(data, rate, 12000, 13000)
    if high <= 0:
        return None if low <= 0 else 99.0
    if low <= 0:
        return -99.0
    return float(10.0 * np.log10(low / high))


def _correlation(candidate: np.ndarray, source: np.ndarray, times: np.ndarray,
                 high_mask: np.ndarray | None = None) -> dict:
    both = np.isfinite(candidate) & np.isfinite(source)
    if high_mask is not None:
        both &= high_mask
    if both.sum() < 3:
        both = np.isfinite(candidate) & np.isfinite(source)
    if both.sum() < 3:
        return {"f0_contour_correlation": None, "octave_error_ratio": None}
    a = _semitone(candidate[both])
    b = _semitone(source[both])
    correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 1.0
    octave_error = float(np.mean(np.abs(a - b) > 6.0))
    return {"f0_contour_correlation": round(correlation, 6),
            "octave_error_ratio": round(octave_error, 6)}


def analyze(candidate, rate: int | None = None, source=None, source_rate: int | None = None,
            *, high_hz: float = HIGH_NOTE_HZ, gate_mode: str = "warn") -> dict:
    """Measure one candidate, optionally against a time-aligned source."""
    if isinstance(candidate, (str, Path)):
        candidate, rate = _load(candidate)
    elif rate is None:
        raise ValueError("rate is required for an in-memory candidate")
    candidate = _mono(candidate)
    rate = int(rate)
    if len(candidate) < rate // 4:
        raise ValueError("Audio is too short to analyze")

    f0, voiced, times = _contour(candidate, rate)
    frame_rate = 1.0 / float(times[1] - times[0])
    voiced_values = f0[np.isfinite(f0)]
    if len(voiced_values) == 0:
        raise ValueError("No voiced frames found")

    high_mask = np.isfinite(f0) & (f0 >= high_hz)
    runs = _segments(high_mask, times, MIN_HIGH_SEGMENT_SECONDS)
    segments = []
    for start, end in runs:
        segment_audio = _resample(candidate, rate, F0_SAMPLE_RATE)[
            int(times[start] * F0_SAMPLE_RATE): int(times[end - 1] * F0_SAMPLE_RATE)]
        depth, vibrato_rate = _vibrato(f0[start:end], frame_rate)
        segments.append({
            "start_s": round(float(times[start]), 3),
            "end_s": round(float(times[end - 1]), 3),
            "duration_s": round(float(times[end - 1] - times[start]), 3),
            "median_f0_hz": round(float(np.nanmedian(f0[start:end])), 1),
            "vibrato_depth_cents": round(depth, 1) if depth is not None else None,
            "vibrato_rate_hz": round(vibrato_rate, 2) if vibrato_rate is not None else None,
            "hnr_db": round(h, 2) if (h := _hnr(segment_audio, F0_SAMPLE_RATE)) is not None else None,
            "harmonic_to_inharmonic_db": round(h, 2) if (
                h := _harmonic_ratio(segment_audio, F0_SAMPLE_RATE, f0[start:end], frame_rate)) is not None else None,
            "spectral_flatness": round(f, 5) if (f := _flatness(segment_audio, F0_SAMPLE_RATE)) is not None else None,
        })

    long_segments = [s for s in segments if s["duration_s"] >= MIN_VIBRATO_SEGMENT_SECONDS
                     and s["vibrato_depth_cents"] is not None]
    flat_time = sum(s["duration_s"] for s in long_segments if s["vibrato_depth_cents"] < FLAT_VIBRATO_CENTS)
    total_long = sum(s["duration_s"] for s in long_segments)
    flat_ratio = float(flat_time / total_long) if total_long else None

    low_mask = np.isfinite(f0) & (f0 >= 200.0) & (f0 <= 400.0)
    low_runs = _segments(low_mask, times, 0.35)
    low_ratio = None
    if low_runs:
        start, end = low_runs[0]
        low_audio = _resample(candidate, rate, F0_SAMPLE_RATE)[
            int(times[start] * F0_SAMPLE_RATE): int(times[end - 1] * F0_SAMPLE_RATE)]
        low_ratio = _harmonic_ratio(low_audio, F0_SAMPLE_RATE, f0[start:end], frame_rate)

    whole = {
        "hf_share_11k": round(hf_share(candidate, rate, 11000, rate / 2), 8),
        "hf_share_12k": round(hf_share(candidate, rate, 12000, rate / 2), 8),
        "hf_share_12_16k": round(hf_share(candidate, rate, 12000, 16000), 8),
        "cliff_12k_db": round(c, 2) if (c := _cliff_db(candidate, rate)) is not None else None,
        "high_note_seconds": round(float(sum(s["duration_s"] for s in segments)), 3),
        "flat_vibrato_ratio": round(flat_ratio, 4) if flat_ratio is not None else None,
        "harmonic_to_inharmonic_low_db": round(low_ratio, 2) if low_ratio is not None else None,
    }

    comparison = None
    if source is not None:
        if isinstance(source, (str, Path)):
            source, source_rate = _load(source)
        elif source_rate is None:
            raise ValueError("source_rate is required for an in-memory source")
        source = _mono(source)
        src_f0, _, src_times = _contour(source, int(source_rate))
        if len(src_f0) != len(f0):
            src_f0 = np.interp(times, src_times, np.nan_to_num(src_f0, nan=0.0))
            src_f0[src_f0 <= 0] = np.nan
        # Correlation over all voiced frames is stable even when a window has
        # only a fraction of a second above the high-note threshold. The
        # high-only correlation is reported separately when enough frames exist.
        comparison = _correlation(f0, src_f0, times, None)
        if int(high_mask.sum()) >= 25:
            high_only = _correlation(f0, src_f0, times, high_mask)
            comparison["high_f0_contour_correlation"] = high_only["f0_contour_correlation"]
            comparison["high_octave_error_ratio"] = high_only["octave_error_ratio"]
        source_hf = hf_share(source, int(source_rate), 11000, int(source_rate) / 2)
        comparison["hf_share_11k_source"] = round(source_hf, 8)
        comparison["hf_ratio_11k"] = round(whole["hf_share_11k"] / source_hf, 4) if source_hf > 0 else None

    flags = []
    if rate <= 24000:
        flags.append("band_limited_24k")
    elif whole["hf_share_12_16k"] < 0.001 and (whole["cliff_12k_db"] or 0) > 20:
        flags.append("band_limited_24k")
    if flat_ratio is not None and flat_ratio > 0.5 and whole["high_note_seconds"] >= 0.5:
        flags.append("flat_vibrato_high_note")
    if comparison and comparison.get("f0_contour_correlation") is not None:
        # 1% octave frames is within tracking noise; 02's real failure is 29%.
        if comparison["f0_contour_correlation"] < 0.97 or (comparison.get("octave_error_ratio") or 0) > 0.02:
            flags.append("pitch_tracking_mismatch")
    if (whole["harmonic_to_inharmonic_low_db"] is not None
            and segments and segments[0]["harmonic_to_inharmonic_db"] is not None
            and segments[0]["harmonic_to_inharmonic_db"] < whole["harmonic_to_inharmonic_low_db"] - 3.0):
        flags.append("inharmonic_hf_excess")

    return {
        "rate": rate,
        "duration_s": round(len(candidate) / rate, 3),
        "f0": {
            "method": "pyin_24k",
            "p50_hz": round(float(np.percentile(voiced_values, 50)), 1),
            "p90_hz": round(float(np.percentile(voiced_values, 90)), 1),
            "p95_hz": round(float(np.percentile(voiced_values, 95)), 1),
            "max_hz": round(float(np.max(voiced_values)), 1),
        },
        "high_segments": segments,
        "whole": whole,
        "comparison": comparison,
        "flags": flags,
        "gate_mode": gate_mode,
        "calibration_status": "synthetic_validated_user_timestamps_pending",
        "note": "Descriptive measurements only; they do not grade musicality or singing quality.",
    }


def self_test() -> dict:
    """Synthetic signals must separate before the metrics are used on real audio."""
    rate = 44100
    seconds = 2.0
    t = np.arange(int(rate * seconds)) / rate
    f0 = 880.0

    def tone(vibrato_cents: float) -> np.ndarray:
        # Peak frequency deviation = f0 * (2**(cents/1200) - 1); the phase
        # modulation index that produces it is deviation / modulation rate.
        deviation = f0 * (2 ** (vibrato_cents / 1200.0) - 1.0)
        phase = 2 * np.pi * f0 * t + (deviation / 5.5) * np.sin(2 * np.pi * 5.5 * t)
        return (0.3 * np.sin(phase) + 0.12 * np.sin(2 * phase) + 0.06 * np.sin(3 * phase))

    with_vibrato = analyze(tone(30.0), rate)
    flat = analyze(tone(0.0), rate)
    band_limited = analyze(_resample(tone(30.0), rate, 24000), 24000)

    assert with_vibrato["high_segments"], "vibrato tone was not detected as a high note"
    assert with_vibrato["whole"]["high_note_seconds"] > 0.5
    vibrato_depth = with_vibrato["high_segments"][0]["vibrato_depth_cents"]
    assert vibrato_depth is not None and vibrato_depth > 15, vibrato_depth
    assert "flat_vibrato_high_note" in flat["flags"], flat["flags"]
    assert "flat_vibrato_high_note" not in with_vibrato["flags"], with_vibrato["flags"]
    assert "band_limited_24k" in band_limited["flags"], band_limited["flags"]
    assert band_limited["whole"]["hf_share_12_16k"] == 0.0
    return {
        "status": "passed",
        "vibrato_depth_cents": vibrato_depth,
        "flat_ratio_vibrato": with_vibrato["whole"]["flat_vibrato_ratio"],
        "flat_ratio_constant": flat["whole"]["flat_vibrato_ratio"],
        "band_limited_flags": band_limited["flags"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="High-note / melisma evidence (no quality claims)")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--source", type=Path, help="Time-aligned reference vocal stem")
    parser.add_argument("--out", type=Path, help="Write JSON here")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--high-hz", type=float, default=HIGH_NOTE_HZ)
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return 0
    if not args.candidate:
        parser.error("--candidate or --self-test is required")
    result = analyze(args.candidate, source=args.source)
    result["file"] = str(args.candidate)
    if args.source:
        result["source"] = str(args.source)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
