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
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "review_queue"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vocal_repair"

FILTERS = {
    "transparent": (
        "adeclick,"
        "adeclip,"
        "deesser=i=0.12:m=0.30:f=0.5,"
        "alimiter=limit=0.97"
    ),
    "review_fast": (
        "deesser=i=0.10:m=0.28:f=0.48,"
        "alimiter=limit=0.97"
    ),
    "gentle": (
        "afftdn=nf=-31,"
        "adeclick,"
        "adeclip,"
        "deesser=i=0.18:m=0.38:f=0.5,"
        "compand=attacks=0.05:decays=0.35:points=-80/-80|-35/-34|-18/-17|0/-1:soft-knee=8:gain=0,"
        "alimiter=limit=0.96"
    ),
    "strong": (
        "highpass=f=45,"
        "lowpass=f=14500,"
        "afftdn=nf=-23,"
        "adeclick,"
        "adeclip,"
        "deesser=i=0.32:m=0.55:f=0.55,"
        "compand=attacks=0.03:decays=0.22:points=-80/-80|-40/-35|-24/-20|-10/-8|0/-1.2:soft-knee=8:gain=1.5,"
        "alimiter=limit=0.93,"
        "loudnorm=I=-16:TP=-1.8:LRA=9"
    ),
}


@dataclass
class VocalAnalysis:
    file: str
    duration_seconds: float
    peak: float
    rms: float
    clipping_ratio: float
    low_energy_gap_seconds: float
    sudden_drop_count: int
    harsh_frame_ratio: float
    artifact_score: int
    status: str
    reasons: list[str]


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")
    return ffmpeg


def _ffprobe() -> str:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found in PATH")
    return ffprobe


def audio_duration(path: Path) -> float:
    output = subprocess.check_output(
        [
            _ffprobe(),
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


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100)
    return ordered[index]


def analyze_file(path: Path, sample_rate: int = 16000) -> VocalAnalysis:
    samples = decode_mono(path, sample_rate=sample_rate)
    if not samples:
        return VocalAnalysis(
            file=str(path),
            duration_seconds=0,
            peak=0,
            rms=0,
            clipping_ratio=0,
            low_energy_gap_seconds=0,
            sudden_drop_count=0,
            harsh_frame_ratio=0,
            artifact_score=100,
            status="needs_regen",
            reasons=["decoded audio is empty"],
        )

    max_int = 32768.0
    duration = audio_duration(path)
    peak = max(abs(v) for v in samples) / max_int
    rms = math.sqrt(sum((v / max_int) ** 2 for v in samples) / len(samples))
    clipping_ratio = sum(1 for v in samples if abs(v) >= 32112) / len(samples)

    frame_size = max(1, round(sample_rate * 0.05))
    frame_rms: list[float] = []
    frame_zcr: list[float] = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        frame_rms.append(math.sqrt(sum((v / max_int) ** 2 for v in frame) / len(frame)))
        crossings = 0
        previous = frame[0]
        for current in frame[1:]:
            if (previous < 0 <= current) or (previous >= 0 > current):
                crossings += 1
            previous = current
        frame_zcr.append(crossings / max(1, len(frame) - 1))

    active_floor = max(0.012, _percentile(frame_rms, 35) * 0.75)
    dropout_threshold = active_floor * 0.35
    low_energy_gap_frames = 0
    sudden_drop_count = 0
    harsh_frames = 0
    active_frames = 0

    for index, value in enumerate(frame_rms):
        prev_window = frame_rms[max(0, index - 4) : index]
        next_window = frame_rms[index + 1 : index + 5]
        prev_active = prev_window and sum(prev_window) / len(prev_window) > active_floor
        next_active = next_window and sum(next_window) / len(next_window) > active_floor
        if value < dropout_threshold and prev_active and next_active:
            low_energy_gap_frames += 1

        if index > 0:
            previous = max(frame_rms[index - 1], 1e-6)
            current = max(value, 1e-6)
            if previous > active_floor and current < previous * 0.18:
                sudden_drop_count += 1

        if value > active_floor:
            active_frames += 1
            if frame_zcr[index] > 0.22:
                harsh_frames += 1

    low_energy_gap_seconds = low_energy_gap_frames * 0.05
    harsh_frame_ratio = harsh_frames / active_frames if active_frames else 0
    artifact_score = round(
        min(
            100,
            low_energy_gap_seconds * 5
            + sudden_drop_count * 0.8
            + clipping_ratio * 1500
            + harsh_frame_ratio * 35,
        )
    )

    reasons: list[str] = []
    if low_energy_gap_seconds > 1.2:
        reasons.append("possible vocal dropouts or broken phrases")
    if sudden_drop_count > 12:
        reasons.append("frequent abrupt level drops")
    if clipping_ratio > 0.002:
        reasons.append("clipping risk")
    if harsh_frame_ratio > 0.12:
        reasons.append("harsh high-frequency/noisy vocal frames")
    if not reasons:
        reasons.append("no hard vocal artifact detected by proxy analysis")

    if artifact_score >= 40:
        status = "needs_regen"
    elif artifact_score >= 16:
        status = "repair_recommended"
    else:
        status = "pass"

    return VocalAnalysis(
        file=str(path),
        duration_seconds=round(duration, 2),
        peak=round(peak, 4),
        rms=round(rms, 4),
        clipping_ratio=round(clipping_ratio, 5),
        low_energy_gap_seconds=round(low_energy_gap_seconds, 2),
        sudden_drop_count=sudden_drop_count,
        harsh_frame_ratio=round(harsh_frame_ratio, 5),
        artifact_score=artifact_score,
        status=status,
        reasons=reasons,
    )


def repair_file(path: Path, output_path: Path, profile: str = "gentle") -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = FILTERS[profile]
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-i",
            str(path),
            "-af",
            filters,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_path


def process_directory(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    profile: str = "transparent",
    repair_all: bool = True,
) -> tuple[Path, Path]:
    wavs = sorted(input_dir.glob("*.wav"))
    output_dir.mkdir(parents=True, exist_ok=True)
    repaired_dir = output_dir / "fullmix_experimental"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows: list[VocalAnalysis] = []
    for wav in wavs:
        analysis = analyze_file(wav)
        rows.append(analysis)
        if repair_all:
            repair_file(wav, repaired_dir / wav.name, profile=profile)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = reports_dir / f"vocal_repair_report_{stamp}.json"
    csv_path = reports_dir / f"vocal_repair_report_{stamp}.csv"
    json_path.write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "file",
                "duration_seconds",
                "artifact_score",
                "status",
                "low_energy_gap_seconds",
                "sudden_drop_count",
                "harsh_frame_ratio",
                "clipping_ratio",
                "reasons",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.file,
                    row.duration_seconds,
                    row.artifact_score,
                    row.status,
                    row.low_energy_gap_seconds,
                    row.sudden_drop_count,
                    row.harsh_frame_ratio,
                    row.clipping_ratio,
                    "；".join(row.reasons),
                ]
            )
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze and repair vocal artifacts in generated songs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile", choices=sorted(FILTERS), default="transparent")
    parser.add_argument(
        "--write-fullmix",
        action="store_true",
        help="Write experimental full-mix repaired WAVs. Default only writes reports.",
    )
    args = parser.parse_args()

    json_path, csv_path = process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        profile=args.profile,
        repair_all=args.write_fullmix,
    )
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    if args.write_fullmix:
        print(f"Repaired audio: {args.output_dir / 'fullmix_experimental'}")
    else:
        print("Full-mix repair not written. Use --write-fullmix only for diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
