from __future__ import annotations

import argparse
import array
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from vocal_repair import PROJECT_ROOT, _ffmpeg


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vocal_repair" / "noise_agent"

VOCAL_FILTERS = {
    "transparent": "adeclick,adeclip,deesser=i=0.10:m=0.25:f=0.45,afftdn=nf=-38,alimiter=limit=0.97",
    "review_fast": "deesser=i=0.10:m=0.25:f=0.45,afftdn=nf=-38,alimiter=limit=0.97",
    "balanced": "highpass=f=55,lowpass=f=15000,adeclick,adeclip,deesser=i=0.16:m=0.36:f=0.5,afftdn=nf=-34,alimiter=limit=0.96",
    "aggressive": "highpass=f=70,lowpass=f=13500,adeclick,adeclip,deesser=i=0.24:m=0.48:f=0.55,afftdn=nf=-29,alimiter=limit=0.95",
}

INSTRUMENTAL_FILTERS = {
    "transparent": "adeclick,adeclip,afftdn=nf=-40,alimiter=limit=0.98",
    "review_fast": "passthrough",
    "balanced": "highpass=f=28,lowpass=f=16500,adeclick,adeclip,afftdn=nf=-36,alimiter=limit=0.97",
    "aggressive": "highpass=f=35,lowpass=f=15000,adeclick,adeclip,afftdn=nf=-32,alimiter=limit=0.96",
}


@dataclass
class NoiseStats:
    file: str
    peak: float
    rms: float
    clipping_ratio: float
    silence_ratio: float
    harsh_frame_ratio: float
    burst_count: int


@dataclass
class NoiseCleanResult:
    input: str
    output: str
    role: str
    profile: str
    filter_chain: str
    before: NoiseStats
    after: NoiseStats


def decode_mono(path: Path, sample_rate: int = 16000) -> array.array:
    raw = subprocess.check_output(
        [
            _ffmpeg(),
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ],
        stderr=subprocess.DEVNULL,
    )
    samples = array.array("h")
    samples.frombytes(raw)
    return samples


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100)
    return ordered[index]


def analyze_noise(path: Path, sample_rate: int = 16000) -> NoiseStats:
    samples = decode_mono(path, sample_rate=sample_rate)
    if not samples:
        return NoiseStats(str(path), 0, 0, 0, 0, 0, 0)

    max_int = 32768.0
    peak = max(abs(sample) for sample in samples) / max_int
    rms = math.sqrt(sum((sample / max_int) ** 2 for sample in samples) / len(samples))
    clipping_ratio = sum(1 for sample in samples if abs(sample) >= 32112) / len(samples)
    silence_ratio = sum(1 for sample in samples if abs(sample) < 328) / len(samples)

    frame_size = max(1, round(sample_rate * 0.05))
    frame_rms: list[float] = []
    frame_zcr: list[float] = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        frame_rms.append(math.sqrt(sum((value / max_int) ** 2 for value in frame) / len(frame)))
        crossings = 0
        previous = frame[0]
        for current in frame[1:]:
            if (previous < 0 <= current) or (previous >= 0 > current):
                crossings += 1
            previous = current
        frame_zcr.append(crossings / max(1, len(frame) - 1))

    active_floor = max(0.01, _percentile(frame_rms, 35) * 0.7)
    active_frames = 0
    harsh_frames = 0
    burst_count = 0
    median_rms = max(_percentile(frame_rms, 50), 1e-6)
    for index, value in enumerate(frame_rms):
        if value > active_floor:
            active_frames += 1
            if frame_zcr[index] > 0.34:
                harsh_frames += 1
        if value > median_rms * 4.5 and frame_zcr[index] > 0.28:
            burst_count += 1

    harsh_frame_ratio = harsh_frames / active_frames if active_frames else 0
    return NoiseStats(
        file=str(path),
        peak=round(peak, 4),
        rms=round(rms, 4),
        clipping_ratio=round(clipping_ratio, 5),
        silence_ratio=round(silence_ratio, 5),
        harsh_frame_ratio=round(harsh_frame_ratio, 5),
        burst_count=burst_count,
    )


def _filters_for(role: str, profile: str) -> str:
    if role == "vocals":
        return VOCAL_FILTERS[profile]
    if role in {"instrumental", "no_vocals"}:
        return INSTRUMENTAL_FILTERS[profile]
    raise ValueError(f"Unknown stem role: {role}")


def clean_stem(
    input_wav: Path,
    output_wav: Path,
    role: str,
    profile: str = "transparent",
    timeout_seconds: int | None = None,
) -> NoiseCleanResult:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    filters = _filters_for(role, profile)
    before = analyze_noise(input_wav)
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-i",
            str(input_wav),
            "-af",
            filters,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_seconds,
    )
    after = analyze_noise(output_wav)
    return NoiseCleanResult(str(input_wav), str(output_wav), role, profile, filters, before, after)


def clean_stems(vocals: Path, instrumental: Path, output_dir: Path, profile: str = "transparent") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    vocal_result = clean_stem(vocals, output_dir / "vocals_noise_clean.wav", "vocals", profile)
    instrumental_output = output_dir / "no_vocals_noise_clean.wav"
    instrumental_before = analyze_noise(instrumental)
    shutil.copy2(instrumental, instrumental_output)
    instrumental_after = analyze_noise(instrumental_output)
    instrumental_result = NoiseCleanResult(
        str(instrumental),
        str(instrumental_output),
        "instrumental",
        profile,
        "passthrough",
        instrumental_before,
        instrumental_after,
    )
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profile": profile,
        "vocals": asdict(vocal_result),
        "instrumental": asdict(instrumental_result),
    }
    (output_dir / "noise_clean_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean non-musical noise from separated vocal/instrumental stems.")
    parser.add_argument("--vocals", type=Path, required=True)
    parser.add_argument("--instrumental", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile", choices=sorted(VOCAL_FILTERS), default="transparent")
    args = parser.parse_args()

    report = clean_stems(args.vocals, args.instrumental, args.output_dir, args.profile)
    print(f"Vocals: {report['vocals']['output']}")
    print(f"Instrumental: {report['instrumental']['output']}")
    print(f"Report: {args.output_dir / 'noise_clean_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
