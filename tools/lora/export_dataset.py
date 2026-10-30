"""Export curated AI_Music songs into ACE-Step LoRA dataset format."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "data" / "app.db"
DEFAULT_TOP_SONGS = PROJECT_ROOT / "outputs" / "top_songs"
DEFAULT_OUT = PROJECT_ROOT / "data" / "lora_datasets" / "from_top_songs"


@dataclass
class Sample:
    audio_path: str
    lyrics: str
    caption: str
    title: str = ""
    language: str = "zh"
    score: int | None = None
    job_id: str | None = None
    source: str = ""


def parse_top_song_txt(text: str) -> tuple[str, int | None, str]:
    """Return (title, score, lyrics) from top_songs sidecar txt."""
    lines = text.splitlines()
    title = ""
    score: int | None = None
    body: list[str] = []
    in_body = False
    for line in lines:
        if not in_body:
            if line.startswith("Title:"):
                title = line.split(":", 1)[1].strip()
                continue
            if line.startswith("Score:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    score = int(float(raw))
                except ValueError:
                    score = None
                continue
            if line.startswith("Category:"):
                continue
            if line.strip() == "":
                in_body = True
                continue
            if line.startswith("["):
                in_body = True
                body.append(line)
                continue
        else:
            body.append(line)
    lyrics = "\n".join(body).strip()
    return title, score, lyrics


def _job_id_from_stem(stem: str) -> str | None:
    # stem: 001_95_review_category_title_jobid
    parts = stem.split("_")
    if not parts:
        return None
    candidate = parts[-1]
    if re.fullmatch(r"[0-9a-fA-F-]{8,}", candidate):
        return candidate
    return candidate if len(candidate) >= 8 else None


def load_jobs_index(db_path: Path) -> dict[str, dict]:
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("pragma table_info(jobs)").fetchall()}
        if "id" not in columns:
            return {}
        rows = conn.execute(
            """
            select id, title, lyrics, style_prompt, audio_path, instrumental
            from jobs
            where status = 'succeeded' and audio_path is not null
            """
        ).fetchall()
    index: dict[str, dict] = {}
    for row in rows:
        job = {
            "id": row[0],
            "title": row[1] or "",
            "lyrics": row[2] or "",
            "style_prompt": row[3] or "",
            "audio_path": row[4] or "",
            "instrumental": bool(row[5]),
        }
        index[job["id"]] = job
        # also index by short prefix for truncated stems
        if len(job["id"]) >= 12:
            index[job["id"][:12]] = job
    return index


def collect_from_top_songs(
    top_dir: Path,
    db_path: Path,
    min_score: int = 0,
) -> list[Sample]:
    jobs = load_jobs_index(db_path)
    samples: list[Sample] = []
    for wav in sorted(top_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        title, score, lyrics = ("", None, "")
        if txt.exists():
            title, score, lyrics = parse_top_song_txt(txt.read_text(encoding="utf-8"))
        if score is not None and score < min_score:
            continue
        job_id = _job_id_from_stem(wav.stem)
        job = None
        if job_id:
            job = jobs.get(job_id) or next(
                (v for k, v in jobs.items() if k.startswith(job_id) or job_id.startswith(k[:12])),
                None,
            )
        if not lyrics and job:
            lyrics = job["lyrics"]
        if not title and job:
            title = job["title"]
        caption = (job or {}).get("style_prompt") or "Chinese pop, realistic female vocal, clean mix"
        if not lyrics.strip():
            continue
        samples.append(
            Sample(
                audio_path=str(wav.resolve()),
                lyrics=lyrics.strip(),
                caption=caption.strip(),
                title=title,
                language="zh",
                score=score,
                job_id=(job or {}).get("id") if job else job_id,
                source="top_songs",
            )
        )
    return samples


def collect_from_db(db_path: Path, min_score: int, limit: int) -> list[Sample]:
    """Fallback: use succeeded jobs; optional score filter via curation if available."""
    from curation_bridge import score_jobs  # local helper

    return score_jobs(db_path, min_score=min_score, limit=limit)


def materialize_folder(samples: list[Sample], out_dir: Path, copy_audio: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    dataset: list[dict] = []
    for index, sample in enumerate(samples, start=1):
        stem = f"{index:03d}_{_safe(sample.title or sample.job_id or 'track')}"
        src = Path(sample.audio_path)
        dst_audio = audio_dir / f"{stem}{src.suffix or '.wav'}"
        if copy_audio:
            shutil.copy2(src, dst_audio)
            audio_path = str(dst_audio.resolve())
        else:
            audio_path = str(src.resolve())
        (audio_dir / f"{stem}.lyrics.txt").write_text(sample.lyrics, encoding="utf-8")
        meta = {
            "caption": sample.caption,
            "language": sample.language,
            "title": sample.title,
            "score": sample.score,
            "job_id": sample.job_id,
        }
        (audio_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        dataset.append(
            {
                "audio_path": audio_path,
                "lyrics": sample.lyrics,
                "caption": sample.caption,
                "language": sample.language,
                "custom_tag": sample.title,
            }
        )
    dataset_path = out_dir / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(dataset),
        "samples": [asdict(s) for s in samples],
        "dataset_json": str(dataset_path.resolve()),
        "audio_dir": str(audio_dir.resolve()),
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset_path


def _safe(name: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", name, flags=re.UNICODE).strip("_")
    return (cleaned or "track")[:40]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ACE-Step LoRA dataset from curated songs.")
    parser.add_argument("--top-songs-dir", type=Path, default=DEFAULT_TOP_SONGS)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-score", type=int, default=85)
    parser.add_argument("--from-db", action="store_true", help="Build from SQLite jobs via curation scores")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--no-copy-audio", action="store_true")
    args = parser.parse_args()

    if args.from_db:
        samples = collect_from_db(args.db, min_score=args.min_score, limit=args.limit)
    else:
        samples = collect_from_top_songs(args.top_songs_dir, args.db, min_score=args.min_score)
        if not samples:
            print("No top_songs samples; falling back to DB scoring.")
            samples = collect_from_db(args.db, min_score=args.min_score, limit=args.limit)

    if not samples:
        print("No samples found. Run top_song_workflow or lower --min-score.")
        return 1

    dataset_path = materialize_folder(samples, args.out_dir, copy_audio=not args.no_copy_audio)
    print(f"Exported {len(samples)} samples")
    print(f"dataset.json: {dataset_path}")
    print(f"audio dir: {args.out_dir / 'audio'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
