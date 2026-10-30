"""Register-aware reference clips for the fixed-voice SVC path.

`iron_rain_female_v1` froze one 7 s reference at a median 319 Hz and its full
vocal stem covered only the first 60 s of the anchor (p90 494 Hz). The anchor's
real high notes are at 81-82.6 s, 103.7 s and 156.1 s (up to 1095 Hz), so the
voice encoder was asked to reproduce a register it had never heard. That is one
of the reasons high notes collapsed into a mechanical timbre.

This module takes a full-band vocal stem of the *whole* anchor and picks low,
mid and high reference clips. Each SVC chunk can then be paired with the clip
whose median F0 is closest to the chunk's own register. The profile files are
written separately so the frozen v1 profile is never modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_RATE = 24000
CONTOUR_RATE = 50.0
HARVEST_PERIOD_MS = 10.0
F0_FLOOR = 65.0
F0_CEIL = 1600.0
MIN_VOICED_RATIO = 0.50
MIN_RMS = 0.005


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _mono(data: np.ndarray) -> np.ndarray:
    value = np.asarray(data, dtype=np.float64)
    if value.ndim == 2:
        value = value.mean(axis=1)
    if value.ndim != 1:
        raise ValueError("Expected mono or (frames, channels) audio")
    if not np.isfinite(value).all():
        raise ValueError("Audio contains NaN or infinite samples")
    return value


def reject_octave_slips(f0: np.ndarray) -> np.ndarray:
    """Reject frames more than 4.5 semitones from >=3 agreeing neighbours."""
    value = np.asarray(f0, dtype=np.float64).copy()
    voiced = value > 0
    if voiced.sum() <= 6:
        return value
    for index in np.where(voiced)[0]:
        neighbours = [value[j] for j in range(max(0, index - 2), min(len(value), index + 3))
                      if j != index and value[j] > 0]
        if len(neighbours) >= 3:
            centre = float(np.median(neighbours))
            if abs(12.0 * np.log2(value[index] / (centre + 1e-9))) > 4.5:
                value[index] = 0.0
    return value


def extract_contour(vocal: Path, rate: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (mono samples, 50 Hz F0, sample rate).

    pyin is used instead of pyworld harvest: in this environment harvest
    reports zero voiced frames on clean 220/440 Hz tones (verified), which would
    silently discard entire registers. pyin at a 480-sample hop is exactly 50 Hz.
    """
    import librosa
    import soundfile as sf
    data, source_rate = sf.read(str(vocal), dtype="float64", always_2d=True)
    mono = _mono(data)
    if source_rate != MODEL_RATE:
        from math import gcd
        from scipy.signal import resample_poly
        common = gcd(int(source_rate), MODEL_RATE)
        analysis = resample_poly(mono, MODEL_RATE // common, source_rate // common)
    else:
        analysis = mono
    contour, voiced, probability = librosa.pyin(
        analysis.astype(np.float32), fmin=F0_FLOOR, fmax=F0_CEIL,
        sr=MODEL_RATE, frame_length=2048, hop_length=480)
    contour = np.where(np.asarray(voiced, dtype=bool), contour, 0.0)
    contour = reject_octave_slips(contour)
    frames = int(np.ceil(len(mono) / source_rate * CONTOUR_RATE))
    contour = np.pad(contour, (0, max(0, frames - len(contour))))[:frames]
    return mono, contour.astype(np.float64), int(source_rate)


def _rms_frames(data: np.ndarray, rate: int) -> np.ndarray:
    hop = max(1, rate // int(CONTOUR_RATE))
    count = int(np.ceil(len(data) / hop))
    padded = np.pad(data, (0, count * hop - len(data)))
    return np.sqrt(np.mean(padded.reshape(count, hop) ** 2, axis=1) + 1e-12)


def _candidates(data: np.ndarray, f0: np.ndarray, rate: int,
                min_seconds: float, max_seconds: float) -> list[dict]:
    levels = _rms_frames(data, rate)
    total = len(f0)
    candidates = []
    for seconds in np.arange(min_seconds, max_seconds + 0.01, 1.0):
        frames = int(round(seconds * CONTOUR_RATE))
        for start in range(0, max(0, total - frames), int(CONTOUR_RATE)):
            window = f0[start:start + frames]
            if len(window) < frames:
                continue
            voiced = window > 0
            ratio = float(np.mean(voiced))
            if ratio < MIN_VOICED_RATIO or voiced.sum() < 3:
                continue
            level = float(np.mean(levels[start:start + frames])) if len(levels) else 0.0
            if level < MIN_RMS:
                continue
            candidates.append(dict(start_frame=start, frames=frames, seconds=float(seconds),
                                   voiced_ratio=ratio, median_f0=float(np.median(window[voiced])),
                                   rms=level))
    return candidates


def _pick(candidates: list[dict], target: float, chosen: list[dict], min_gap_frames: int) -> dict | None:
    ordered = sorted(candidates, key=lambda c: abs(np.log(c["median_f0"] / target)))
    for candidate in ordered:
        if all(abs(candidate["start_frame"] - old["start_frame"]) >= min_gap_frames for old in chosen):
            return candidate
    return None


def extract_registers(vocal: Path, out_dir: Path, *, min_seconds: float = 3.0,
                      max_seconds: float = 8.0) -> dict:
    """Write low/mid/high reference clips and a registers.json manifest."""
    import soundfile as sf
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mono, f0, rate = extract_contour(vocal)
    candidates = _candidates(mono, f0, rate, min_seconds, max_seconds)
    if not candidates:
        raise ValueError("No active, stable reference candidates found")
    medians = np.array([c["median_f0"] for c in candidates])
    targets = {"low": float(np.percentile(medians, 25)),
               "mid": float(np.percentile(medians, 50)),
               # The high register is a small tail: p90 of all candidate
               # medians can still sit below 500 Hz. p99 tracks the actual
               # highest sustained passage instead.
               "high": float(np.percentile(medians, 99))}
    min_gap = int(round(min_seconds * CONTOUR_RATE))
    chosen: dict[str, dict] = {}
    picked: list[dict] = []
    for name in ("low", "mid", "high"):
        candidate = _pick(candidates, targets[name], picked, min_gap)
        if candidate is None:
            continue
        chosen[name] = candidate
        picked.append(candidate)

    registers = {}
    for name, candidate in chosen.items():
        start = int(round(candidate["start_frame"] / CONTOUR_RATE * rate))
        end = min(len(mono), start + int(round(candidate["seconds"] * rate)))
        clip = mono[start:end]
        full_path = out_dir / f"{name}_full.wav"
        prompt_path = out_dir / f"{name}.wav"
        sf.write(str(full_path), clip, rate, subtype="PCM_16")
        from math import gcd
        from scipy.signal import resample_poly
        common = gcd(int(rate), MODEL_RATE)
        prompt = resample_poly(clip, MODEL_RATE // common, rate // common)
        sf.write(str(prompt_path), prompt, MODEL_RATE, subtype="PCM_16")
        window = f0[candidate["start_frame"]:candidate["start_frame"] + candidate["frames"]]
        contour_path = out_dir / f"{name}_f0.npy"
        np.save(contour_path, window.astype("float32"))
        registers[name] = {
            "full": {"path": str(full_path), "sha256": digest(full_path), "rate": rate},
            "prompt": {"path": str(prompt_path), "sha256": digest(prompt_path), "rate": MODEL_RATE},
            "f0": {"path": str(contour_path), "sha256": digest(contour_path)},
            "start_seconds": round(candidate["start_frame"] / CONTOUR_RATE, 3),
            "seconds": round(candidate["seconds"], 3),
            "voiced_ratio": round(candidate["voiced_ratio"], 4),
            "median_f0_hz": round(candidate["median_f0"], 1),
        }

    high_median = registers.get("high", {}).get("median_f0_hz")
    manifest = {
        "schema": 1,
        "source": str(Path(vocal).resolve()),
        "source_sha256": digest(Path(vocal)),
        "source_rate": rate,
        "contour_rate": CONTOUR_RATE,
        "f0_method": "librosa_pyin_24k_hop480_50hz_octave_rejected",
        "target_medians_hz": {name: round(value, 1) for name, value in targets.items()},
        "registers": registers,
        "register_gap": None if high_median is not None and high_median >= 600.0 else "anchor_lacks_high_register",
        "note": "Register clips are prompts for SVC; the frozen v1 reference is not modified.",
    }
    (out_dir / "registers.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def select_register(manifest: dict, target_f0: float) -> str | None:
    """Return the register name whose median F0 is closest in log-F0."""
    registers = manifest.get("registers", {})
    if not registers or target_f0 <= 0:
        return next(iter(registers), None)
    return min(registers, key=lambda name: abs(
        np.log(max(target_f0, 1.0) / max(registers[name]["median_f0_hz"], 1.0))))
