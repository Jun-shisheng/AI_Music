from __future__ import annotations

import argparse
import array
import csv
import json
import math
import re
import shutil
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "data" / "app.db"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "curation"

MAINSTREAM_STYLE_TERMS = {
    "pop",
    "mandopop",
    "chinese-style",
    "guofeng",
    "ballad",
    "r&b",
    "folk",
    "healing",
    "dreamy",
    "emotional",
    "piano",
    "guitar",
    "strings",
    "female vocals",
    "male vocals",
}

NICHE_STYLE_TERMS = {
    "noise",
    "atonal",
    "experimental",
    "industrial",
    "glitch",
    "sfx",
    "horror",
}

CONCRETE_IMAGES = {
    "雨",
    "风",
    "海",
    "夜",
    "星",
    "月",
    "街",
    "灯",
    "窗",
    "雪",
    "花",
    "river",
    "rain",
    "street",
    "night",
    "moon",
    "light",
    "window",
}


@dataclass
class Track:
    id: str
    title: str
    lyrics: str
    style_prompt: str
    instrumental: bool
    generation_type: str
    audio_path: str
    created_at: str


@dataclass
class AudioStats:
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    peak: float | None = None
    rms: float | None = None
    clipping_ratio: float | None = None
    silence_ratio: float | None = None
    noise_floor: float | None = None
    dynamic_range: float | None = None
    error: str | None = None


@dataclass
class ScoreResult:
    id: str
    title: str
    lyrics: str
    total: int
    selected: bool
    decision: str
    reasons: list[str]
    dimensions: dict[str, int]
    audio_stats: AudioStats
    style_prompt: str
    audio_path: str


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_tracks(db_path: Path = DEFAULT_DB, limit: int = 100) -> list[Track]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("pragma table_info(jobs)").fetchall()}
        generation_expr = "generation_type" if "generation_type" in columns else "'song'"
        duration_expr = "duration_seconds" if "duration_seconds" in columns else "null"
        query = f"""
            select id, title, lyrics, style_prompt, instrumental, {generation_expr}, audio_path, created_at, {duration_expr}
            from jobs
            where status = 'succeeded' and audio_path is not null
            order by created_at desc
            limit ?
        """
        rows = conn.execute(query, (limit,)).fetchall()

    tracks: list[Track] = []
    for row in rows:
        tracks.append(
            Track(
                id=row[0],
                title=row[1] or "Untitled",
                lyrics=row[2] or "",
                style_prompt=row[3] or "",
                instrumental=bool(row[4]),
                generation_type=row[5] or "song",
                audio_path=row[6] or "",
                created_at=row[7] or "",
            )
        )
    return tracks


def read_audio_stats(audio_path: Path, sample_limit: int = 1_500_000) -> AudioStats:
    if not audio_path.exists():
        return AudioStats(error=f"missing audio file: {audio_path}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return AudioStats(duration_seconds=_probe_duration(audio_path), error="ffmpeg not found")

    sample_rate = 16000
    max_seconds = max(1, sample_limit // sample_rate)
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-t",
        str(max_seconds),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except Exception as exc:  # noqa: BLE001
        return AudioStats(duration_seconds=_probe_duration(audio_path), error=f"ffmpeg decode failed: {exc}")

    samples = array.array("h")
    samples.frombytes(raw)
    if not samples:
        return AudioStats(duration_seconds=_probe_duration(audio_path), error="decoded audio is empty")

    values = [sample / 32768.0 for sample in samples]
    peak = max(abs(v) for v in values)
    rms = math.sqrt(sum(v * v for v in values) / len(values))
    clipping = sum(1 for v in values if abs(v) >= 0.98) / len(values)
    silence = sum(1 for v in values if abs(v) < 0.01) / len(values)
    chunk_rms = _chunk_rms(values, max(1, sample_rate // 5))

    return AudioStats(
        duration_seconds=_probe_duration(audio_path),
        sample_rate=sample_rate,
        channels=1,
        peak=round(peak, 4),
        rms=round(rms, 4),
        clipping_ratio=round(clipping, 5),
        silence_ratio=round(silence, 5),
        noise_floor=_percentile(chunk_rms, 10),
        dynamic_range=_dynamic_range(chunk_rms),
    )


def _probe_duration(audio_path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        return round(float(output), 2)
    except Exception:  # noqa: BLE001
        return None


def _chunk_rms(values: list[float], chunk_size: int) -> list[float]:
    chunks: list[float] = []
    for index in range(0, len(values), chunk_size):
        chunk = values[index : index + chunk_size]
        if chunk:
            chunks.append(math.sqrt(sum(v * v for v in chunk) / len(chunk)))
    return chunks


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100)
    return round(ordered[index], 4)


def _dynamic_range(values: list[float]) -> float | None:
    low = _percentile(values, 10)
    high = _percentile(values, 90)
    if low is None or high is None or low <= 0:
        return None
    return round(high / low, 2)


def _section_names(lyrics: str) -> list[str]:
    return [m.group(1).strip().lower() for m in re.finditer(r"\[([^\]]+)\]", lyrics)]


def _clean_lines(lyrics: str) -> list[str]:
    lines = []
    for line in lyrics.splitlines():
        cleaned = re.sub(r"\[[^\]]+\]", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _section_blocks(lyrics: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current = "_preamble"
    for raw in lyrics.splitlines():
        tag = re.match(r"^\s*\[([^\]]+)\]\s*$", raw.strip())
        if tag:
            current = tag.group(1).strip().lower()
            blocks.setdefault(current, [])
            continue
        cleaned = re.sub(r"\[[^\]]+\]", "", raw).strip()
        if cleaned:
            blocks.setdefault(current, []).append(cleaned)
    return blocks


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _fatigue_signals(lyrics: str) -> tuple[float, float, list[str]]:
    """Return (unique_ratio, verse_similarity, reason strings) for scoring penalties."""
    lines = _clean_lines(lyrics)
    reasons: list[str] = []
    if not lines:
        return 1.0, 0.0, reasons
    unique_ratio = len(set(lines)) / len(lines)
    blocks = _section_blocks(lyrics)
    verses = [
        blines
        for label, blines in blocks.items()
        if blines and ("主歌" in label or label.startswith("verse"))
    ]
    verse_sim = 0.0
    if len(verses) >= 2:
        verse_sim = max(
            _jaccard(verses[i], verses[j])
            for i in range(len(verses))
            for j in range(i + 1, len(verses))
        )
    non_chorus: list[str] = []
    for label, blines in blocks.items():
        if "副歌" in label or "chorus" in label:
            continue
        non_chorus.extend(blines)
    non_chorus_unique = len(set(non_chorus)) / len(non_chorus) if non_chorus else 1.0
    if unique_ratio < 0.55:
        reasons.append(f"歌词疲劳重复 unique_ratio={unique_ratio:.2f}")
    if non_chorus_unique < 0.75:
        reasons.append(f"非副歌复读过高 unique={non_chorus_unique:.2f}")
    if verse_sim >= 0.6:
        reasons.append(f"主歌相似度过高 similarity={verse_sim:.2f}")
    return unique_ratio, verse_sim, reasons


def score_track(track: Track) -> ScoreResult:
    audio_path = _resolve_path(track.audio_path)
    stats = read_audio_stats(audio_path)
    sections = _section_names(track.lyrics)
    lines = _clean_lines(track.lyrics)
    style = track.style_prompt.lower()
    is_instrumental = track.instrumental or not lines or "[instrumental]" in track.lyrics.lower()
    reasons: list[str] = []

    narrative = 0
    if is_instrumental:
        narrative = 35
        reasons.append("纯音乐，单独进入纯音乐队列")
    else:
        if len(lines) >= 24:
            narrative += 25
        elif len(lines) >= 16:
            narrative += 20
        elif len(lines) >= 8:
            narrative += 15
        if any(term in "".join(lines).lower() for term in CONCRETE_IMAGES):
            narrative += 25
        has_structure = (
            {"verse", "chorus"}.issubset(set(sections))
            or ("主歌" in track.lyrics and "副歌" in track.lyrics)
            or any("chorus" in s or "副歌" in s for s in sections)
        )
        if has_structure:
            narrative += 25
        unique_ratio = len(set(lines)) / len(lines) if lines else 0
        if 0.7 <= unique_ratio <= 0.98:
            narrative += 25
        elif 0.5 <= unique_ratio < 0.7:
            narrative += 12
        elif unique_ratio < 0.5:
            narrative -= 20
            reasons.append("歌词重复导致叙事疲劳")
        if len(set(sections)) >= 5:
            narrative += 10
        _, verse_sim, fatigue_reasons = _fatigue_signals(track.lyrics)
        if verse_sim >= 0.6:
            narrative -= 15
        for reason in fatigue_reasons:
            if reason not in reasons:
                reasons.append(reason)
    narrative = max(0, min(narrative, 100))

    hook = 0
    chorus_count = sum(1 for s in sections if "chorus" in s or "副歌" in s)
    if chorus_count:
        hook += 35
    if track.title and any(track.title.lower() in line.lower() for line in lines):
        hook += 20
    repeated_lines = len(lines) - len(set(lines))
    # Purposeful chorus hooks: small repeat count is good; wall-to-wall repeats hurt.
    if 1 <= repeated_lines <= 8:
        hook += 25
    elif repeated_lines == 0 and chorus_count:
        hook += 15
    elif repeated_lines > 12:
        hook -= 20
        reasons.append("重复行过多，易造成听感疲劳")
    if any(6 <= len(line) <= 18 for line in lines):
        hook += 20
    hook = max(0, min(hook, 100))

    mainstream = 45
    mainstream += sum(8 for term in MAINSTREAM_STYLE_TERMS if term in style)
    mainstream -= sum(15 for term in NICHE_STYLE_TERMS if term in style)
    tempo_match = re.search(r"(\d{2,3})\s*[-~]\s*(\d{2,3})|(\d{2,3})\s*bpm", style)
    if tempo_match:
        nums = [int(n) for n in tempo_match.groups() if n]
        avg_tempo = sum(nums) / len(nums)
        if 72 <= avg_tempo <= 138:
            mainstream += 15
    mainstream = max(0, min(mainstream, 100))

    production = 50
    if stats.error:
        production = 15
        reasons.append(stats.error)
    else:
        duration = stats.duration_seconds or 0
        if is_instrumental:
            if 60 <= duration <= 240:
                production += 20
        elif 190 <= duration <= 240:
            production += 30
        elif 170 <= duration < 190 or 240 < duration <= 260:
            production += 22
        elif 120 <= duration < 170 or 260 < duration <= 300:
            production += 8
            reasons.append("正式歌曲时长偏离 3 分半目标")
        else:
            reasons.append("正式歌曲时长不合格")

        if (stats.peak or 0) < 0.98 and (stats.clipping_ratio or 0) < 0.002:
            production += 15
        else:
            reasons.append("音频存在削波或过载风险")

        if 0.02 <= (stats.rms or 0) <= 0.35:
            production += 10
        else:
            reasons.append("整体响度异常，需要复核")

        if (stats.noise_floor or 0) > 0.08 and (stats.dynamic_range or 99) < 3:
            production -= 25
            reasons.append("底噪/持续杂音偏高，可能听感不干净")
    production = max(0, min(production, 100))

    platform = 50
    if track.generation_type == "song":
        platform += 15
    if not is_instrumental and lines:
        platform += 15
    if stats.duration_seconds and 180 <= stats.duration_seconds <= 245:
        platform += 15
    elif stats.duration_seconds and stats.duration_seconds < 170 and not is_instrumental:
        platform -= 15
    platform = max(0, min(platform, 100))

    dimensions = {
        "narrative_clarity": narrative,
        "hook_strength": hook,
        "mainstream_fit": mainstream,
        "production_readiness": production,
        "platform_packaging": platform,
    }
    total = round(
        narrative * 0.24
        + hook * 0.24
        + mainstream * 0.2
        + production * 0.2
        + platform * 0.12
    )

    if narrative < 65 and not is_instrumental:
        reasons.append("叙事/歌词变化不足，正式歌曲支撑不够")
    if hook < 55 and not is_instrumental:
        reasons.append("副歌或记忆点偏弱")
    if mainstream < 55:
        reasons.append("风格离主流短视频音乐审美较远")
    if production < 70:
        reasons.append("音频时长、响度、杂音或文件质量需要复核")

    selected = total >= 75 and production >= 70 and mainstream >= 55 and not is_instrumental
    decision = "review_priority" if selected else "hold_for_review"
    if selected:
        reasons.insert(0, "适合作为人工审阅优先曲目")

    return ScoreResult(
        id=track.id,
        title=track.title,
        lyrics=track.lyrics,
        total=total,
        selected=selected,
        decision=decision,
        reasons=reasons,
        dimensions=dimensions,
        audio_stats=stats,
        style_prompt=track.style_prompt,
        audio_path=str(audio_path),
    )


def write_reports(results: list[ScoreResult], report_dir: Path = DEFAULT_REPORT_DIR) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = report_dir / f"curation_report_{stamp}.json"
    csv_path = report_dir / f"curation_report_{stamp}.csv"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_threshold": 70,
        "results": [
            {
                **asdict(result),
                "audio_stats": asdict(result.audio_stats),
            }
            for result in results
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "id", "title", "total", "selected", "decision", "reasons", "audio_path"])
        for index, result in enumerate(results, start=1):
            writer.writerow(
                [
                    index,
                    result.id,
                    result.title,
                    result.total,
                    result.selected,
                    result.decision,
                    "；".join(result.reasons),
                    result.audio_path,
                ]
            )
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Score generated songs for social media publishing.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to app.db")
    parser.add_argument("--limit", type=int, default=100, help="Max succeeded jobs to inspect")
    parser.add_argument("--min-score", type=int, default=70, help="Print review tracks at or above this score")
    args = parser.parse_args()

    tracks = load_tracks(args.db, args.limit)
    results = sorted((score_track(track) for track in tracks), key=lambda item: item.total, reverse=True)
    json_path, csv_path = write_reports(results)

    print(f"Scored {len(results)} tracks")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    print("")
    print("Top candidates:")
    for result in results[:10]:
        marker = "REVIEW" if result.total >= args.min_score else "NEEDS_WORK"
        print(f"{marker} {result.total:3d}  {result.id}  {result.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
