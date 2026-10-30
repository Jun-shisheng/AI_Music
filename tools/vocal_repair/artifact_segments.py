from __future__ import annotations

import argparse
import array
import csv
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "review_queue"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vocal_repair" / "artifact_segments"


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")
    return ffmpeg


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


def detect_events(
    path: Path,
    sample_rate: int = 16000,
    min_dropout_severity: float = 0.7,
    min_drop_ratio: float = 8.0,
    min_noise_zcr: float = 0.38,
    max_events_per_file: int = 12,
) -> list[dict]:
    samples = decode_mono(path, sample_rate)
    max_int = 32768.0
    frame_seconds = 0.05
    frame_size = max(1, round(sample_rate * frame_seconds))
    rms_values: list[float] = []
    zcr_values: list[float] = []

    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        rms_values.append(math.sqrt(sum((value / max_int) ** 2 for value in frame) / len(frame)))
        crossings = 0
        previous = frame[0]
        for current in frame[1:]:
            if (previous < 0 <= current) or (previous >= 0 > current):
                crossings += 1
            previous = current
        zcr_values.append(crossings / max(1, len(frame) - 1))

    active_floor = max(0.012, _percentile(rms_values, 35) * 0.75)
    dropout_threshold = active_floor * 0.35
    events: list[dict] = []

    for index, value in enumerate(rms_values):
        timestamp = round(index * frame_seconds, 2)
        prev_window = rms_values[max(0, index - 4) : index]
        next_window = rms_values[index + 1 : index + 5]
        prev_active = prev_window and sum(prev_window) / len(prev_window) > active_floor
        next_active = next_window and sum(next_window) / len(next_window) > active_floor

        if value < dropout_threshold and prev_active and next_active:
            severity = round((dropout_threshold - value) / max(dropout_threshold, 1e-6), 3)
            if severity < min_dropout_severity:
                continue
            events.append(
                {
                    "file": str(path),
                    "time_seconds": timestamp,
                    "type": "possible_dropout",
                    "severity": severity,
                    "detail": "short low-energy gap between active regions; check for broken word or phrase",
                }
            )

        if index > 0:
            previous = max(rms_values[index - 1], 1e-6)
            current = max(value, 1e-6)
            drop_ratio = round(previous / current, 2)
            if previous > active_floor and current < previous * 0.18 and drop_ratio >= min_drop_ratio:
                events.append(
                    {
                        "file": str(path),
                        "time_seconds": timestamp,
                        "type": "abrupt_level_drop",
                        "severity": drop_ratio,
                        "detail": "sudden level collapse; check for AI vocal break",
                    }
                )

        if value > active_floor and zcr_values[index] > min_noise_zcr:
            events.append(
                {
                    "file": str(path),
                    "time_seconds": timestamp,
                    "type": "harsh_machine_noise",
                    "severity": round(zcr_values[index], 3),
                    "detail": "high zero-crossing active frame; check for robotic noise or brittle high note",
                }
            )

    events.sort(key=lambda event: float(event["severity"]), reverse=True)
    return sorted(events[:max_events_per_file], key=lambda event: (event["file"], event["time_seconds"]))


def write_reports(
    input_dir: Path,
    output_dir: Path,
    min_dropout_severity: float = 0.7,
    min_drop_ratio: float = 8.0,
    min_noise_zcr: float = 0.38,
    max_events_per_file: int = 12,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    for wav in sorted(input_dir.glob("*.wav")):
        events.extend(
            detect_events(
                wav,
                min_dropout_severity=min_dropout_severity,
                min_drop_ratio=min_drop_ratio,
                min_noise_zcr=min_noise_zcr,
                max_events_per_file=max_events_per_file,
            )
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"artifact_segments_{stamp}.json"
    csv_path = output_dir / f"artifact_segments_{stamp}.csv"
    json_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "time_seconds", "type", "severity", "detail"])
        writer.writeheader()
        writer.writerows(events)
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Report likely AI vocal breaks and machine-noise timestamps.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-dropout-severity", type=float, default=0.7)
    parser.add_argument("--min-drop-ratio", type=float, default=8.0)
    parser.add_argument("--min-noise-zcr", type=float, default=0.38)
    parser.add_argument("--max-events-per-file", type=int, default=12)
    args = parser.parse_args()

    json_path, csv_path = write_reports(
        args.input_dir,
        args.output_dir,
        min_dropout_severity=args.min_dropout_severity,
        min_drop_ratio=args.min_drop_ratio,
        min_noise_zcr=args.min_noise_zcr,
        max_events_per_file=args.max_events_per_file,
    )
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
