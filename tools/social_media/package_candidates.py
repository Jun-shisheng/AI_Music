from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from curation import DEFAULT_DB, PROJECT_ROOT, load_tracks, score_track


DEFAULT_PACKAGE_DIR = PROJECT_ROOT / "outputs" / "publish_candidates"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "outputs" / "review_queue"
DEFAULT_NEEDS_WORK_DIR = PROJECT_ROOT / "outputs" / "needs_work_queue"
DEFAULT_INSTRUMENTAL_DIR = PROJECT_ROOT / "outputs" / "instrumental_queue"


def clean_queue_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.wav", "*.json"):
        for item in path.glob(pattern):
            item.unlink()


def safe_name(text: str, fallback: str = "untitled", max_len: int = 48) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("._ ")
    return (cleaned or fallback)[:max_len]


def category_for(style_prompt: str) -> str:
    style = style_prompt.lower()
    if any(term in style for term in ["battle", "anime rock", "pop rock", "metal", "epic", "phonk"]):
        return "battle"
    if any(term in style for term in ["chinese", "guzheng", "erhu", "pipa", "guofeng", "国风"]):
        return "guofeng"
    if any(term in style for term in ["female", "sweet", "healing", "dreamy", "piano"]):
        return "sweet_female"
    return "mixed"


def is_instrumental_result(result) -> bool:
    title = result.title.lower()
    style = result.style_prompt.lower()
    reasons = " ".join(result.reasons)
    return (
        "instrumental" in title
        or "instrumental" in style
        or "纯音乐" in reasons
        or "no vocals" in style
    )


def has_quality_blocker(result) -> bool:
    reasons = " ".join(result.reasons)
    return (
        bool(result.audio_stats.error)
        or result.dimensions["production_readiness"] < 70
        or "底噪" in reasons
        or "持续杂音" in reasons
        or "削波" in reasons
        or "过载" in reasons
        or "响度异常" in reasons
    )


def package_candidates(
    db_path: Path = DEFAULT_DB,
    review_dir: Path = DEFAULT_REVIEW_DIR,
    needs_work_dir: Path = DEFAULT_NEEDS_WORK_DIR,
    instrumental_dir: Path = DEFAULT_INSTRUMENTAL_DIR,
    min_score: int = 70,
    needs_work_score: int = 45,
    limit: int = 200,
    copy_audio: bool = True,
    clean: bool = False,
    track_ids: set[str] | None = None,
    max_review: int | None = None,
) -> list[Path]:
    tracks = load_tracks(db_path, limit)
    if track_ids is not None:
        tracks = [track for track in tracks if track.id in track_ids]
    scored = sorted((score_track(track) for track in tracks), key=lambda item: item.total, reverse=True)
    if clean:
        clean_queue_dir(review_dir)
        clean_queue_dir(needs_work_dir)
        clean_queue_dir(instrumental_dir)
    else:
        review_dir.mkdir(parents=True, exist_ok=True)
        needs_work_dir.mkdir(parents=True, exist_ok=True)
        instrumental_dir.mkdir(parents=True, exist_ok=True)

    review_manifest = []
    needs_work_manifest = []
    instrumental_manifest = []
    written: list[Path] = []
    review_count = 0
    for rank, result in enumerate(scored, start=1):
        if is_instrumental_result(result):
            queue = "instrumental"
            target_dir = instrumental_dir
        elif has_quality_blocker(result):
            queue = "needs_work"
            target_dir = needs_work_dir
        elif result.total >= min_score:
            queue = "review"
            target_dir = review_dir
            if max_review is not None and review_count >= max_review:
                queue = "needs_work"
                target_dir = needs_work_dir
        elif result.total >= needs_work_score or result.audio_stats.error or result.dimensions["production_readiness"] < 65:
            queue = "needs_work"
            target_dir = needs_work_dir
        else:
            continue

        category = category_for(result.style_prompt)
        stem = f"{rank:03d}_{result.total}_{queue}_{category}_{safe_name(result.title)}_{result.id}"

        audio_src = Path(result.audio_path)
        audio_dst = target_dir / f"{stem}{audio_src.suffix or '.wav'}"
        if copy_audio and audio_src.exists():
            shutil.copy2(audio_src, audio_dst)
            written.append(audio_dst)

        sidecar = target_dir / f"{stem}.json"
        payload = {
            "rank": rank,
            "category": category,
            "queue": queue,
            "review_name": stem if queue == "review" else None,
            "needs_work_name": stem if queue == "needs_work" else None,
            "instrumental_name": stem if queue == "instrumental" else None,
            "score": {
                **asdict(result),
                "audio_stats": asdict(result.audio_stats),
            },
            "audio": str(audio_dst) if copy_audio else None,
        }
        sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(sidecar)
        if queue == "review":
            review_count += 1
            review_manifest.append(payload)
        elif queue == "needs_work":
            needs_work_manifest.append(payload)
        else:
            instrumental_manifest.append(payload)

    review_manifest_path = review_dir / "manifest.json"
    review_manifest_path.write_text(json.dumps(review_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(review_manifest_path)
    needs_work_manifest_path = needs_work_dir / "manifest.json"
    needs_work_manifest_path.write_text(json.dumps(needs_work_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(needs_work_manifest_path)
    instrumental_manifest_path = instrumental_dir / "manifest.json"
    instrumental_manifest_path.write_text(json.dumps(instrumental_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(instrumental_manifest_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy and rename publishable songs into review folders.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to app.db")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR, help="Human review queue folder")
    parser.add_argument("--needs-work-dir", type=Path, default=DEFAULT_NEEDS_WORK_DIR, help="Needs-work queue folder")
    parser.add_argument("--instrumental-dir", type=Path, default=DEFAULT_INSTRUMENTAL_DIR, help="Instrumental queue folder")
    parser.add_argument("--min-score", type=int, default=70, help="Minimum score for human review")
    parser.add_argument("--needs-work-score", type=int, default=45, help="Minimum score for the needs-work queue")
    parser.add_argument("--limit", type=int, default=200, help="Max succeeded jobs to inspect")
    parser.add_argument("--no-audio-copy", action="store_true", help="Only write metadata sidecars")
    parser.add_argument("--clean", action="store_true", help="Clear old queue WAV/JSON files before packaging")
    parser.add_argument("--max-review", type=int, default=None, help="Only put the top N passing songs into human review")
    args = parser.parse_args()

    written = package_candidates(
        db_path=args.db,
        review_dir=args.review_dir,
        needs_work_dir=args.needs_work_dir,
        instrumental_dir=args.instrumental_dir,
        min_score=args.min_score,
        needs_work_score=args.needs_work_score,
        limit=args.limit,
        copy_audio=not args.no_audio_copy,
        clean=args.clean,
        max_review=args.max_review,
    )
    print(f"Packaged files: {len(written)}")
    print(f"Review folder: {args.review_dir}")
    print(f"Needs-work folder: {args.needs_work_dir}")
    print(f"Instrumental folder: {args.instrumental_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
