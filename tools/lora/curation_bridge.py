"""Score SQLite jobs for LoRA export using social_media curation heuristics."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOCIAL = PROJECT_ROOT / "tools" / "social_media"
if str(SOCIAL) not in sys.path:
    sys.path.insert(0, str(SOCIAL))

from curation import load_tracks, score_track  # noqa: E402

from export_dataset import Sample  # noqa: E402


def score_jobs(db_path: Path, min_score: int = 85, limit: int = 50) -> list[Sample]:
    tracks = load_tracks(db_path=db_path, limit=max(limit * 3, 100))
    scored = sorted((score_track(track) for track in tracks), key=lambda item: item.total, reverse=True)
    samples: list[Sample] = []
    for result in scored:
        if result.total < min_score:
            continue
        if not result.lyrics.strip():
            continue
        audio = Path(result.audio_path)
        if not audio.is_absolute():
            audio = PROJECT_ROOT / audio
        if not audio.exists():
            continue
        samples.append(
            Sample(
                audio_path=str(audio.resolve()),
                lyrics=result.lyrics.strip(),
                caption=(result.style_prompt or "Chinese pop, realistic vocal").strip(),
                title=result.title,
                language="zh",
                score=result.total,
                job_id=result.id,
                source="db_curation",
            )
        )
        if len(samples) >= limit:
            break
    return samples
