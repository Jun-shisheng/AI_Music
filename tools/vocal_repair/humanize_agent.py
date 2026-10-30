from __future__ import annotations

import argparse
import array
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from vocal_repair import PROJECT_ROOT, _ffmpeg


DEFAULT_REFS_DIR = PROJECT_ROOT / "human_refs" / "vocals"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vocal_repair" / "humanize_agent"


@dataclass
class VocalReferenceStats:
    file: str
    duration_seconds: float
    rms: float
    peak: float
    dynamic_range: float
    silence_ratio: float
    harsh_frame_ratio: float


@dataclass
class HumanizeProfile:
    refs_dir: str
    reference_count: int
    median_rms: float
    median_dynamic_range: float
    median_harsh_frame_ratio: float
    filter_chain: str


_PROFILE_CACHE: dict[str, HumanizeProfile] = {}


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


def duration_seconds(path: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    return float(output)


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * pct / 100)]


def analyze_reference(path: Path, sample_rate: int = 16000) -> VocalReferenceStats:
    samples = decode_mono(path, sample_rate=sample_rate)
    if not samples:
        return VocalReferenceStats(str(path), 0, 0, 0, 0, 1, 0)

    max_int = 32768.0
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

    peak = max(abs(sample) for sample in samples) / max_int
    rms = math.sqrt(sum((sample / max_int) ** 2 for sample in samples) / len(samples))
    active_floor = max(0.008, percentile(frame_rms, 35) * 0.65)
    active = [value for value in frame_rms if value > active_floor]
    dynamic_range = percentile(active, 90) - percentile(active, 25) if active else 0
    silence_ratio = sum(1 for value in frame_rms if value <= active_floor * 0.4) / max(1, len(frame_rms))
    harsh_frames = sum(
        1
        for index, value in enumerate(frame_rms)
        if value > active_floor and index < len(frame_zcr) and frame_zcr[index] > 0.32
    )
    harsh_frame_ratio = harsh_frames / max(1, len(active))
    return VocalReferenceStats(
        file=str(path),
        duration_seconds=round(duration_seconds(path), 2),
        rms=round(rms, 5),
        peak=round(peak, 5),
        dynamic_range=round(dynamic_range, 5),
        silence_ratio=round(silence_ratio, 5),
        harsh_frame_ratio=round(harsh_frame_ratio, 5),
    )


def build_profile(refs_dir: Path = DEFAULT_REFS_DIR) -> HumanizeProfile:
    cache_key = str(refs_dir.resolve())
    if cache_key in _PROFILE_CACHE:
        return _PROFILE_CACHE[cache_key]

    refs = sorted(refs_dir.rglob("*.wav"))
    stats = [analyze_reference(ref) for ref in refs]
    useful = [item for item in stats if item.duration_seconds >= 30 and item.rms > 0.002]
    if not useful:
        filter_chain = "adeclick,adeclip,deesser=i=0.10:m=0.28:f=0.48,alimiter=limit=0.97"
        profile = HumanizeProfile(str(refs_dir), 0, 0, 0, 0, filter_chain)
        _PROFILE_CACHE[cache_key] = profile
        return profile

    median_rms = percentile([item.rms for item in useful], 50)
    median_dynamic_range = percentile([item.dynamic_range for item in useful], 50)
    median_harsh = percentile([item.harsh_frame_ratio for item in useful], 50)

    high_soften_db = -0.8 if median_harsh < 0.08 else -1.4
    presence_soften_db = -0.6 if median_dynamic_range > 0.05 else -1.0
    filter_chain = (
        f"equalizer=f=3100:t=q:w=1.15:g={presence_soften_db:.1f},"
        f"equalizer=f=8200:t=q:w=1.10:g={high_soften_db:.1f},"
        "alimiter=limit=0.97"
    )
    profile = HumanizeProfile(
        refs_dir=str(refs_dir),
        reference_count=len(useful),
        median_rms=round(median_rms, 5),
        median_dynamic_range=round(median_dynamic_range, 5),
        median_harsh_frame_ratio=round(median_harsh, 5),
        filter_chain=filter_chain,
    )
    _PROFILE_CACHE[cache_key] = profile
    return profile


def humanize_vocals(input_wav: Path, output_wav: Path, refs_dir: Path = DEFAULT_REFS_DIR) -> dict:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    profile = build_profile(refs_dir)
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-i",
            str(input_wav),
            "-af",
            profile.filter_chain,
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
    )
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_wav),
        "output": str(output_wav),
        "profile": asdict(profile),
    }
    (output_wav.parent / "humanize_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply reference-driven humanization to a vocal stem.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refs-dir", type=Path, default=DEFAULT_REFS_DIR)
    args = parser.parse_args()
    report = humanize_vocals(args.input, args.output, refs_dir=args.refs_dir)
    print(f"Humanized vocals: {report['output']}")
    print(f"References used: {report['profile']['reference_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
