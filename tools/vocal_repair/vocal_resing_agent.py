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

from artifact_segments import detect_events
from rvc_convert import convert_vocals, is_rvc_configured
from stem_pipeline import separate_with_demucs
from vocal_repair import PROJECT_ROOT, _ffmpeg, analyze_file as analyze_vocal_artifacts, audio_duration


DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "top_songs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vocal_resing"


@dataclass
class VocalSegment:
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    event_count: int
    event_types: list[str]
    action: str
    vocal_clip: str | None = None
    resung_clip: str | None = None
    rvc_status: str | None = None
    rvc_error: str | None = None


@dataclass
class SongResingPlan:
    input: str
    song_dir: str
    vocals: str
    instrumental: str
    artifact_score: int
    artifact_status: str
    decision: str
    reasons: list[str]
    segments: list[VocalSegment]
    review_mix: str | None = None


def _decode_mono(path: Path, sample_rate: int = 16000) -> array.array:
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
    return ordered[round((len(ordered) - 1) * percentile / 100)]


def _merge_intervals(intervals: list[tuple[float, float]], max_gap: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def detect_vocal_phrases(
    vocals: Path,
    *,
    sample_rate: int = 16000,
    frame_seconds: float = 0.10,
    min_phrase_seconds: float = 0.75,
    merge_gap_seconds: float = 0.45,
    pad_seconds: float = 0.12,
) -> list[tuple[float, float]]:
    samples = _decode_mono(vocals, sample_rate=sample_rate)
    if not samples:
        return []

    max_int = 32768.0
    frame_size = max(1, round(sample_rate * frame_seconds))
    rms_values: list[float] = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        rms_values.append(math.sqrt(sum((value / max_int) ** 2 for value in frame) / len(frame)))

    active_floor = max(0.006, _percentile(rms_values, 45) * 1.15)
    raw_intervals: list[tuple[float, float]] = []
    active_start: float | None = None
    for index, value in enumerate(rms_values):
        timestamp = index * frame_seconds
        if value >= active_floor and active_start is None:
            active_start = timestamp
        elif value < active_floor and active_start is not None:
            raw_intervals.append((active_start, timestamp))
            active_start = None
    if active_start is not None:
        raw_intervals.append((active_start, len(rms_values) * frame_seconds))

    duration = audio_duration(vocals)
    padded = [
        (max(0.0, start - pad_seconds), min(duration, end + pad_seconds))
        for start, end in raw_intervals
        if end - start >= min_phrase_seconds
    ]
    return _merge_intervals(padded, merge_gap_seconds)


def _events_in_segment(events: list[dict], start: float, end: float) -> list[dict]:
    return [event for event in events if start <= float(event["time_seconds"]) <= end]


def _segment_action(event_count: int, decision: str) -> str:
    if decision == "full_resing":
        return "resing_required"
    if event_count:
        return "resing_required"
    if decision == "repair_only":
        return "repair_or_keep"
    return "keep"


def _cut_audio(input_wav: Path, output_wav: Path, start: float, end: float) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(input_wav),
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
    return output_wav


def _assemble_review_mix(
    instrumental: Path,
    vocals: Path,
    segments: list[VocalSegment],
    output_wav: Path,
) -> Path | None:
    replacements = [segment for segment in segments if segment.resung_clip and Path(segment.resung_clip).exists()]
    if not replacements:
        return None

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    inputs = ["-i", str(instrumental), "-i", str(vocals)]
    for segment in replacements:
        inputs.extend(["-i", str(segment.resung_clip)])

    vocal_filters = ["[1:a]"]
    for segment in replacements:
        vocal_filters.append(
            f"volume=enable='between(t,{segment.start_seconds:.3f},{segment.end_seconds:.3f})':volume=0"
        )
    vocal_filters.append("[basev]")
    filter_parts = ["".join(vocal_filters)]

    mix_inputs = ["[basev]"]
    for offset, segment in enumerate(replacements, start=2):
        delay_ms = max(0, round(segment.start_seconds * 1000))
        label = f"rv{offset}"
        filter_parts.append(f"[{offset}:a]adelay={delay_ms}|{delay_ms}[{label}]")
        mix_inputs.append(f"[{label}]")
    filter_parts.append(
        "".join(mix_inputs)
        + f"amix=inputs={len(mix_inputs)}:duration=longest:dropout_transition=0,"
        "alimiter=limit=0.96[vocalmix]"
    )
    filter_parts.append(
        "[0:a][vocalmix]amix=inputs=2:duration=longest:dropout_transition=0,"
        "alimiter=limit=0.95,loudnorm=I=-16:TP=-1.5:LRA=11[out]"
    )
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[out]",
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
    return output_wav


def _decision(artifact_score: int, bad_segment_count: int, resing_score: int, full_resing_score: int) -> str:
    if artifact_score >= full_resing_score or bad_segment_count >= 5:
        return "full_resing"
    if artifact_score >= resing_score or bad_segment_count:
        return "partial_resing"
    return "repair_only"


def build_resing_plan(
    input_wav: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    force: bool = False,
    demucs_model: str = "htdemucs",
    resing_score: int = 16,
    full_resing_score: int = 40,
    execute_rvc: bool = False,
    max_rvc_segments: int = 0,
    review_dir: Path | None = None,
    rvc_model: str | None = None,
    rvc_index: str | None = None,
    rvc_infer_cmd: str | None = None,
) -> SongResingPlan:
    input_wav = Path(input_wav)
    song_dir = output_dir / input_wav.stem
    clips_dir = song_dir / "vocal_segments"
    resung_dir = song_dir / "resung_segments"
    if force and song_dir.exists():
        shutil.rmtree(song_dir)
    song_dir.mkdir(parents=True, exist_ok=True)

    vocals, instrumental = separate_with_demucs(input_wav, song_dir, model=demucs_model)
    artifact = analyze_vocal_artifacts(vocals)
    events = detect_events(vocals, max_events_per_file=40)
    phrases = detect_vocal_phrases(vocals)
    if not phrases:
        duration = audio_duration(vocals)
        phrases = [(0.0, duration)]

    preliminary_events = [_events_in_segment(events, start, end) for start, end in phrases]
    bad_segment_count = sum(1 for segment_events in preliminary_events if segment_events)
    decision = _decision(artifact.artifact_score, bad_segment_count, resing_score, full_resing_score)

    segments: list[VocalSegment] = []
    executed_rvc_count = 0
    for index, ((start, end), segment_events) in enumerate(zip(phrases, preliminary_events), start=1):
        action = _segment_action(len(segment_events), decision)
        clip = clips_dir / f"{index:03d}_{start:.2f}_{end:.2f}.wav"
        _cut_audio(vocals, clip, start, end)
        segment = VocalSegment(
            index=index,
            start_seconds=round(start, 2),
            end_seconds=round(end, 2),
            duration_seconds=round(end - start, 2),
            event_count=len(segment_events),
            event_types=sorted({str(event["type"]) for event in segment_events}),
            action=action,
            vocal_clip=str(clip),
        )
        should_execute_rvc = execute_rvc and action == "resing_required"
        if should_execute_rvc and max_rvc_segments > 0 and executed_rvc_count >= max_rvc_segments:
            should_execute_rvc = False
            segment.rvc_status = "skipped_max_segments"
        if should_execute_rvc:
            rvc_output = resung_dir / clip.name
            executed_rvc_count += 1
            rvc_result = convert_vocals(
                clip,
                rvc_output,
                enabled=True,
                model_path=rvc_model,
                index_path=rvc_index,
                infer_cmd=rvc_infer_cmd,
            )
            segment.rvc_status = rvc_result.status
            segment.rvc_error = rvc_result.error
            if rvc_result.status == "succeeded" and rvc_result.output:
                segment.resung_clip = rvc_result.output
        segments.append(segment)

    reasons = list(artifact.reasons)
    if bad_segment_count:
        reasons.append(f"{bad_segment_count} vocal phrase segments contain detected artifacts")
    if execute_rvc and not is_rvc_configured(enabled=True, model_path=rvc_model, infer_cmd=rvc_infer_cmd):
        reasons.append("RVC execution requested but no converter is configured")

    plan = SongResingPlan(
        input=str(input_wav),
        song_dir=str(song_dir),
        vocals=str(vocals),
        instrumental=str(instrumental),
        artifact_score=artifact.artifact_score,
        artifact_status=artifact.status,
        decision=decision,
        reasons=reasons,
        segments=segments,
    )
    if execute_rvc:
        review_output = song_dir / "review_mix.wav"
        mixed = _assemble_review_mix(instrumental, vocals, segments, review_output)
        if mixed:
            plan.review_mix = str(mixed)
            if review_dir:
                review_dir.mkdir(parents=True, exist_ok=True)
                review_target = review_dir / f"{input_wav.stem}_rvc_segment_review.wav"
                shutil.copy2(mixed, review_target)
                lyrics_file = input_wav.with_suffix(".txt")
                if lyrics_file.exists():
                    shutil.copy2(lyrics_file, review_dir / f"{input_wav.stem}_rvc_segment_review.txt")
                plan.review_mix = str(review_target)
    (song_dir / "resing_manifest.json").write_text(
        json.dumps(asdict(plan), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return plan


def run_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    limit: int = 0,
    force: bool = False,
    execute_rvc: bool = False,
    max_rvc_segments: int = 0,
    review_dir: Path | None = None,
    rvc_model: str | None = None,
    rvc_index: str | None = None,
    rvc_infer_cmd: str | None = None,
) -> list[SongResingPlan]:
    wavs = sorted(Path(input_dir).glob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    plans = [
        build_resing_plan(
            wav,
            output_dir,
            force=force,
            execute_rvc=execute_rvc,
            max_rvc_segments=max_rvc_segments,
            review_dir=review_dir,
            rvc_model=rvc_model,
            rvc_index=rvc_index,
            rvc_infer_cmd=rvc_infer_cmd,
        )
        for wav in wavs
    ]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "items": [asdict(plan) for plan in plans],
    }
    report_path = output_dir / f"resing_report_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    return plans


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan phrase-level vocal re-singing for AI music.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execute-rvc", action="store_true", help="Run configured RVC on segments marked resing_required.")
    parser.add_argument("--max-rvc-segments", type=int, default=0, help="Limit RVC conversion to the first N required segments; 0 means all.")
    parser.add_argument("--review-dir", type=Path, default=None, help="Copy review mixes and lyrics to this folder.")
    parser.add_argument("--rvc-model", default=None)
    parser.add_argument("--rvc-index", default=None)
    parser.add_argument("--rvc-infer-cmd", default=None)
    args = parser.parse_args()

    plans = run_directory(
        args.input_dir,
        args.output_dir,
        limit=args.limit,
        force=args.force,
        execute_rvc=args.execute_rvc,
        max_rvc_segments=args.max_rvc_segments,
        review_dir=args.review_dir,
        rvc_model=args.rvc_model,
        rvc_index=args.rvc_index,
        rvc_infer_cmd=args.rvc_infer_cmd,
    )
    for plan in plans:
        required = sum(1 for segment in plan.segments if segment.action == "resing_required")
        print(
            f"{plan.decision:15s} score={plan.artifact_score:3d} "
            f"resing_segments={required:2d} {Path(plan.input).name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
