from __future__ import annotations

import argparse
import math
import random
import re
import shutil
import sqlite3
import subprocess
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "data" / "app.db"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "social_media"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

TEMPLATE_COLORS = {
    "clean": ("0x111318", "0xE1B96A", "0xFFFFFF"),
    "sweet": ("0x17131F", "0xFF86C8", "0xFFF2FA"),
    "guofeng": ("0x121411", "0xD8A646", "0xF7EFE1"),
    "battle": ("0x090A0F", "0xFF334E", "0xFFFFFF"),
}


def _ass_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds - math.floor(seconds)) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def _resolve_audio_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _duration(audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    except wave.Error:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise
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


def _load_job(job_id: str, db_path: Path) -> tuple[str, str, Path]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "select title, lyrics, audio_path from jobs where id = ? and audio_path is not null",
            (job_id,),
        ).fetchone()
    if not row:
        raise ValueError(f"Job not found or has no audio: {job_id}")
    return row[0] or "Untitled", row[1] or "", _resolve_audio_path(row[2])


def _clean_lyrics(lyrics: str) -> list[str]:
    lines: list[str] = []
    for line in lyrics.splitlines():
        section = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
        if section:
            continue
        cleaned = re.sub(r"\[[^\]]+\]", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _wrap_line(text: str, width: int = 18) -> list[str]:
    if len(text) <= width:
        return [text]
    return [text[i : i + width] for i in range(0, len(text), width)]


def build_ass(title: str, lyrics: str, duration: float, ass_path: Path, template: str = "clean") -> Path:
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for raw in _clean_lyrics(lyrics):
        lines.extend(_wrap_line(raw))
    if not lines:
        lines = ["AI Music Studio", "Instrumental"]

    lyric_text = r"\N".join(_escape_ass(line) for line in lines)
    scroll_start_ms = 600
    scroll_end_ms = max(scroll_start_ms + 1000, int((duration - 1.0) * 1000))
    _, accent, primary = TEMPLATE_COLORS.get(template, TEMPLATE_COLORS["clean"])
    primary_ass = "&H00" + primary.removeprefix("0x")[4:6] + primary.removeprefix("0x")[2:4] + primary.removeprefix("0x")[0:2]
    accent_ass = "&H00" + accent.removeprefix("0x")[4:6] + accent.removeprefix("0x")[2:4] + accent.removeprefix("0x")[0:2]
    ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title, Microsoft YaHei, 62, {accent_ass}, &H000000FF, &H90202020, &H80000000, -1, 0, 0, 0, 100, 100, 0, 0, 1, 2, 1, 8, 80, 80, 118, 1
Style: Lyric, Microsoft YaHei, 50, {primary_ass}, &H000000FF, &H90202020, &HAA000000, -1, 0, 0, 0, 100, 100, 0, 0, 1, 2, 0, 5, 120, 120, 0, 1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 2,{_ass_time(0)},{_ass_time(duration)},Title,,0,0,0,,{{\\fad(300,300)\\blur0.6}}{_escape_ass(title)}
Dialogue: 1,{_ass_time(0)},{_ass_time(duration)},Lyric,,0,0,0,,{{\\move(540,1780,540,-560,{scroll_start_ms},{scroll_end_ms})\\q2\\blur0.4}}{lyric_text}
"""
    ass_path.write_text(ass, encoding="utf-8")
    return ass_path


def _probe_media_duration(media_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    try:
        output = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return float(output)
    except Exception:
        return 0


def _collect_clips(clips_dir: Path) -> list[Path]:
    if not clips_dir.exists():
        raise FileNotFoundError(clips_dir)
    return sorted(path for path in clips_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)


def _write_clip_concat(clips_dir: Path, duration: float, concat_path: Path, seed: int = 7) -> Path:
    clips = _collect_clips(clips_dir)
    if not clips:
        raise ValueError(f"No video clips found in {clips_dir}")
    rng = random.Random(seed)
    lines = ["ffconcat version 1.0"]
    elapsed = 0.0
    while elapsed < duration + 2:
        clip = rng.choice(clips)
        clip_duration = _probe_media_duration(clip)
        if clip_duration <= 2:
            continue
        segment = min(rng.uniform(2.2, 4.8), clip_duration)
        start = rng.uniform(0, max(0.0, clip_duration - segment))
        lines.append(f"file '{clip.as_posix()}'")
        lines.append(f"inpoint {start:.3f}")
        lines.append(f"outpoint {start + segment:.3f}")
        elapsed += segment
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return concat_path


def _procedural_filter(template: str, ass_filter_path: str, accent: str) -> str:
    if template == "battle":
        return (
            "format=yuv420p,"
            "noise=alls=24:allf=t+u,"
            "eq=contrast=1.38:saturation=1.25:brightness=-0.05,"
            "drawbox=x=0:y=0:w=1080:h=1920:color=0x090A0F@0.40:t=fill,"
            f"drawbox=x=42:y=150:w=996:h=6:color={accent}@0.85:t=fill,"
            f"subtitles='{ass_filter_path}'"
        )
    if template == "sweet":
        return (
            "format=yuv420p,"
            "noise=alls=8:allf=t,"
            "eq=contrast=1.08:saturation=1.28:brightness=0.02,"
            f"drawbox=x=76:y=210:w=928:h=4:color={accent}@0.75:t=fill,"
            "vignette=PI/5,"
            f"subtitles='{ass_filter_path}'"
        )
    if template == "guofeng":
        return (
            "format=yuv420p,"
            "noise=alls=10:allf=t,"
            "eq=contrast=1.12:saturation=0.88:brightness=-0.02,"
            f"drawbox=x=76:y=210:w=928:h=4:color={accent}@0.75:t=fill,"
            "vignette=PI/4,"
            f"subtitles='{ass_filter_path}'"
        )
    return (
        "format=yuv420p,"
        f"drawbox=x=76:y=210:w=928:h=4:color={accent}@0.55:t=fill,"
        f"subtitles='{ass_filter_path}'"
    )


def render_video(
    audio_path: Path,
    title: str,
    lyrics: str,
    out_path: Path,
    template: str = "clean",
    clips_dir: Path | None = None,
    verbose: bool = False,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = _duration(audio_path)
    ass_path = out_path.with_suffix(".ass")
    build_ass(title, lyrics, duration, ass_path, template=template)

    ass_filter_path = str(ass_path).replace("\\", "/").replace(":", "\\:")
    base, accent, _ = TEMPLATE_COLORS.get(template, TEMPLATE_COLORS["clean"])
    vf = _procedural_filter(template, ass_filter_path, accent)
    if clips_dir:
        concat_path = out_path.with_suffix(".ffconcat")
        _write_clip_concat(clips_dir, duration, concat_path)
        clip_vf = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            "setsar=1,"
            "fps=30,"
            f"{vf}"
        )
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-i",
            str(audio_path),
            "-t",
            f"{duration:.2f}",
            "-vf",
            clip_vf,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_path),
        ]
    else:
        background = f"color=c={base}:s=1080x1920:r=30:d={duration:.2f}"
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            background,
            "-i",
            str(audio_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_path),
        ]
    subprocess.run(
        cmd,
        check=True,
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.DEVNULL,
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a vertical social-media video with scrolling lyrics.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--job-id", help="Generated job id from the local library")
    source.add_argument("--audio", type=Path, help="Path to a wav audio file")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to app.db")
    parser.add_argument("--title", default=None, help="Video title when using --audio")
    parser.add_argument("--lyrics-file", type=Path, default=None, help="UTF-8 lyrics text file when using --audio")
    parser.add_argument("--out", type=Path, default=None, help="Output mp4 path")
    parser.add_argument(
        "--template",
        choices=["clean", "sweet", "guofeng", "battle"],
        default="clean",
        help="Visual style template",
    )
    parser.add_argument("--clips-dir", type=Path, default=None, help="Folder of legal video clips to auto-cut as background")
    parser.add_argument("--verbose", action="store_true", help="Show ffmpeg logs")
    args = parser.parse_args()

    if args.job_id:
        title, lyrics, audio_path = _load_job(args.job_id, args.db)
        out_path = args.out or (DEFAULT_OUT_DIR / f"{args.job_id}.mp4")
    else:
        audio_path = args.audio
        title = args.title or audio_path.stem
        lyrics = args.lyrics_file.read_text(encoding="utf-8") if args.lyrics_file else ""
        out_path = args.out or (DEFAULT_OUT_DIR / f"{audio_path.stem}.mp4")

    rendered = render_video(
        audio_path,
        title,
        lyrics,
        out_path,
        template=args.template,
        clips_dir=args.clips_dir,
        verbose=args.verbose,
    )
    print(f"Rendered: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
