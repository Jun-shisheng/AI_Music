from __future__ import annotations

import argparse
import array
import csv
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "top_songs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vocal_quality_gate"
DEFAULT_WORK_DIR = PROJECT_ROOT / "outputs" / "vocal_quality_work"


@dataclass
class GateResult:
    file: str
    decision: str
    score: int
    duration_seconds: float
    harsh_frame_ratio: float
    tonal_instability: float
    dropout_seconds: float
    burst_count: int
    reasons: list[str]


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")
    return ffmpeg


def _python() -> str:
    venv_python = PROJECT_ROOT / "tools" / "vocal_repair" / ".venv" / "Scripts" / "python.exe"
    return str(venv_python) if venv_python.exists() else "python"


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


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * pct / 100)]


def separate_vocals(input_wav: Path, work_dir: Path, model: str = "htdemucs") -> Path:
    stem_dir = work_dir / input_wav.stem / "demucs" / model / input_wav.stem
    vocals = stem_dir / "vocals.wav"
    if vocals.exists():
        return vocals
    subprocess.run(
        [
            _python(),
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "-n",
            model,
            "-o",
            str(work_dir / input_wav.stem / "demucs"),
            str(input_wav),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    if not vocals.exists():
        raise FileNotFoundError(f"Demucs did not produce vocals stem: {vocals}")
    return vocals


def analyze_vocals(path: Path, source_file: Path, sample_rate: int = 16000) -> GateResult:
    samples = decode_mono(path, sample_rate=sample_rate)
    max_int = 32768.0
    frame_size = max(1, round(sample_rate * 0.05))
    rms_values: list[float] = []
    zcr_values: list[float] = []
    spectral_flux: list[float] = []
    previous_abs = 0.0

    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        rms = math.sqrt(sum((value / max_int) ** 2 for value in frame) / len(frame))
        rms_values.append(rms)
        crossings = 0
        previous = frame[0]
        abs_mean = sum(abs(value) for value in frame) / (len(frame) * max_int)
        for current in frame[1:]:
            if (previous < 0 <= current) or (previous >= 0 > current):
                crossings += 1
            previous = current
        zcr_values.append(crossings / max(1, len(frame) - 1))
        if previous_abs:
            spectral_flux.append(abs(abs_mean - previous_abs) / max(previous_abs, 1e-6))
        previous_abs = abs_mean

    duration = duration_seconds(source_file)
    active_floor = max(0.01, percentile(rms_values, 35) * 0.7)
    active_indices = [i for i, value in enumerate(rms_values) if value > active_floor]
    harsh_frames = [
        i
        for i in active_indices
        if i < len(zcr_values) and zcr_values[i] > 0.30
    ]
    harsh_frame_ratio = len(harsh_frames) / max(1, len(active_indices))

    dropout_frames = 0
    burst_count = 0
    median_rms = max(percentile(rms_values, 50), 1e-6)
    for index, value in enumerate(rms_values):
        prev_window = rms_values[max(0, index - 4) : index]
        next_window = rms_values[index + 1 : index + 5]
        prev_active = prev_window and sum(prev_window) / len(prev_window) > active_floor
        next_active = next_window and sum(next_window) / len(next_window) > active_floor
        if value < active_floor * 0.25 and prev_active and next_active:
            dropout_frames += 1
        if value > median_rms * 4.0 and index < len(zcr_values) and zcr_values[index] > 0.24:
            burst_count += 1

    tonal_instability = percentile(spectral_flux, 90)
    dropout_seconds = dropout_frames * 0.05

    score = 100
    reasons: list[str] = []
    if harsh_frame_ratio > 0.11:
        penalty = min(30, round((harsh_frame_ratio - 0.11) * 220))
        score -= penalty
        reasons.append(f"high harsh/robotic frame ratio {harsh_frame_ratio:.3f}")
    if tonal_instability > 1.4:
        penalty = min(25, round((tonal_instability - 1.4) * 18))
        score -= penalty
        reasons.append(f"unstable vocal texture {tonal_instability:.2f}")
    if dropout_seconds > 0.35:
        penalty = min(25, round(dropout_seconds * 16))
        score -= penalty
        reasons.append(f"possible broken syllable/dropout {dropout_seconds:.2f}s")
    if burst_count > 2:
        penalty = min(20, burst_count * 3)
        score -= penalty
        reasons.append(f"sudden noisy bursts {burst_count}")

    score = max(0, min(100, score))
    if score >= 82:
        decision = "pass"
    elif score >= 68:
        decision = "manual_review"
    else:
        decision = "reject"
    return GateResult(
        file=str(source_file),
        decision=decision,
        score=score,
        duration_seconds=round(duration, 2),
        harsh_frame_ratio=round(harsh_frame_ratio, 5),
        tonal_instability=round(tonal_instability, 5),
        dropout_seconds=round(dropout_seconds, 2),
        burst_count=burst_count,
        reasons=reasons,
    )


def run_gate(input_dir: Path, output_dir: Path, work_dir: Path, limit: int = 0) -> list[GateResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(input_dir.glob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    results: list[GateResult] = []
    for wav in wavs:
        vocals = separate_vocals(wav, work_dir)
        results.append(analyze_vocals(vocals, wav))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"vocal_quality_gate_{stamp}.json"
    csv_path = output_dir / f"vocal_quality_gate_{stamp}.csv"
    payload = [asdict(result) for result in results]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(payload[0].keys()) if payload else ["file"])
        writer.writeheader()
        writer.writerows(payload)
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate songs by vocal artifact quality.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    results = run_gate(args.input_dir, args.output_dir, args.work_dir, limit=args.limit)
    for result in results:
        print(f"{result.decision:13s} {result.score:3d} {Path(result.file).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
