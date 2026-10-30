from __future__ import annotations

import argparse
import array
import json
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

from render_video import DEFAULT_DB, PROJECT_ROOT, render_video


DEFAULT_SHORT_DIR = PROJECT_ROOT / "outputs" / "social_shorts"


def _resolve_audio_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_job(job_id: str, db_path: Path) -> tuple[str, str, Path]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "select title, lyrics, audio_path from jobs where id = ? and audio_path is not null",
            (job_id,),
        ).fetchone()
    if not row:
        raise ValueError(f"Job not found or has no audio: {job_id}")
    return row[0] or "Untitled", row[1] or "", _resolve_audio_path(row[2])


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")
    return ffmpeg


def _duration(audio_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    output = subprocess.check_output(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    return float(output)


def detect_energy_peak(audio_path: Path, clip_duration: float, sample_rate: int = 8000) -> float:
    cmd = [
        _ffmpeg(),
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    samples = array.array("h")
    samples.frombytes(raw)
    if not samples:
        return 0.0

    chunk_size = max(1, sample_rate // 2)
    energies: list[float] = []
    for index in range(0, len(samples), chunk_size):
        chunk = samples[index : index + chunk_size]
        if not chunk:
            continue
        energies.append(sum(abs(value) for value in chunk) / len(chunk))

    window = max(1, round(clip_duration / 0.5))
    if len(energies) <= window:
        return 0.0

    total_duration = len(samples) / sample_rate
    guard_chunks = max(2, round(4 / 0.5))
    best_index = guard_chunks
    best_score = -1.0
    for index in range(guard_chunks, max(guard_chunks + 1, len(energies) - window - guard_chunks)):
        score = sum(energies[index : index + window])
        if score > best_score:
            best_score = score
            best_index = index

    start = best_index * 0.5
    return max(0.0, min(start, max(0.0, total_duration - clip_duration)))


def _clean_line(line: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", line).strip()


def lyric_excerpt(lyrics: str, max_lines: int = 8) -> str:
    sections: list[tuple[str, list[str]]] = []
    current_name = "body"
    current_lines: list[str] = []
    for raw in lyrics.splitlines():
        section = re.match(r"^\s*\[([^\]]+)\]\s*$", raw)
        if section:
            if current_lines:
                sections.append((current_name, current_lines))
            current_name = section.group(1).lower()
            current_lines = []
            continue
        cleaned = _clean_line(raw)
        if cleaned:
            current_lines.append(cleaned)
    if current_lines:
        sections.append((current_name, current_lines))

    for name, lines in sections:
        if "chorus" in name or "副歌" in name:
            return "\n".join(lines[:max_lines])

    all_lines = [line for _, lines in sections for line in lines]
    if not all_lines:
        return ""
    start = max(0, len(all_lines) // 2 - max_lines // 2)
    return "\n".join(all_lines[start : start + max_lines])


def cut_audio(audio_path: Path, start: float, duration: float, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg(),
        "-y",
        "-ss",
        f"{start:.2f}",
        "-t",
        f"{duration:.2f}",
        "-i",
        str(audio_path),
        "-af",
        "afade=t=in:st=0:d=0.2,afade=t=out:st={:.2f}:d=0.35".format(max(0.0, duration - 0.35)),
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def render_short(
    audio_path: Path,
    title: str,
    lyrics: str,
    out_path: Path,
    clip_duration: float,
    start: float | None = None,
    template: str = "battle",
    clips_dir: Path | None = None,
    verbose: bool = False,
) -> tuple[Path, Path]:
    total_duration = _duration(audio_path)
    if start is None:
        start = detect_energy_peak(audio_path, clip_duration)
        reason = "energy_peak"
    else:
        reason = "manual_start"
    start = max(0.0, min(start, max(0.0, total_duration - clip_duration)))

    audio_clip = out_path.with_suffix(".wav")
    cut_audio(audio_path, start, clip_duration, audio_clip)
    short_lyrics = lyric_excerpt(lyrics)
    rendered = render_video(
        audio_clip,
        f"{title} | {round(clip_duration)}s",
        short_lyrics,
        out_path,
        template=template,
        clips_dir=clips_dir,
        verbose=verbose,
    )
    metadata_path = out_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "source_audio": str(audio_path),
                "output_video": str(rendered),
                "title": title,
                "start_seconds": round(start, 2),
                "duration_seconds": clip_duration,
                "selection_reason": reason,
                "template": template,
                "lyrics": short_lyrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return rendered, metadata_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an 8-30s vertical short from a generated song.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--job-id", help="Generated job id from the local library")
    source.add_argument("--audio", type=Path, help="Path to a wav audio file")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to app.db")
    parser.add_argument("--title", default=None, help="Title when using --audio")
    parser.add_argument("--lyrics-file", type=Path, default=None, help="UTF-8 lyrics text file when using --audio")
    parser.add_argument("--duration", type=float, default=15.0, help="Short duration in seconds")
    parser.add_argument("--start", type=float, default=None, help="Manual start time in seconds")
    parser.add_argument("--template", choices=["clean", "sweet", "guofeng", "battle"], default="battle")
    parser.add_argument("--clips-dir", type=Path, default=None, help="Folder of legal video clips to auto-cut as background")
    parser.add_argument("--out", type=Path, default=None, help="Output mp4 path")
    parser.add_argument("--verbose", action="store_true", help="Show ffmpeg logs")
    args = parser.parse_args()

    if not 5 <= args.duration <= 60:
        raise ValueError("--duration must be between 5 and 60 seconds")

    if args.job_id:
        title, lyrics, audio_path = _load_job(args.job_id, args.db)
        out_path = args.out or (DEFAULT_SHORT_DIR / f"{args.job_id}_{round(args.duration)}s_{args.template}.mp4")
    else:
        audio_path = args.audio
        title = args.title or audio_path.stem
        lyrics = args.lyrics_file.read_text(encoding="utf-8") if args.lyrics_file else ""
        out_path = args.out or (DEFAULT_SHORT_DIR / f"{audio_path.stem}_{round(args.duration)}s_{args.template}.mp4")

    rendered, metadata = render_short(
        audio_path=audio_path,
        title=title,
        lyrics=lyrics,
        out_path=out_path,
        clip_duration=args.duration,
        start=args.start,
        template=args.template,
        clips_dir=args.clips_dir,
        verbose=args.verbose,
    )
    print(f"Rendered short: {rendered}")
    print(f"Metadata: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
