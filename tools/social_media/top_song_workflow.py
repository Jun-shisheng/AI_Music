from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from package_candidates import (
    DEFAULT_INSTRUMENTAL_DIR,
    DEFAULT_NEEDS_WORK_DIR,
    DEFAULT_REVIEW_DIR,
    PROJECT_ROOT,
    package_candidates,
)

DEFAULT_TOP_SONGS_DIR = PROJECT_ROOT / "outputs" / "top_songs"
BATCH_DIR = PROJECT_ROOT / "data" / "formal_batches"
VOCAL_REPAIR_DIR = PROJECT_ROOT / "tools" / "vocal_repair"
VOCAL_REPAIR_VENV = VOCAL_REPAIR_DIR / ".venv" / "Scripts" / "python.exe"
DEFAULT_HUMAN_REFS_DIR = PROJECT_ROOT / "human_refs" / "vocals"


def _python_for_vocal_repair() -> str:
    if VOCAL_REPAIR_VENV.exists():
        return str(VOCAL_REPAIR_VENV)
    return sys.executable


def _clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in {".wav", ".txt", ".json"}:
            item.unlink()


def _remove_work_queue(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _latest_top_batch() -> Path | None:
    batches = [
        *BATCH_DIR.glob("top_batch_*.json"),
        *BATCH_DIR.glob("style100_batch_*.json"),
    ]
    batches = sorted(batches, key=lambda path: path.stat().st_mtime, reverse=True)
    return batches[0] if batches else None


def _load_batch_ids(batch_file: Path | None, all_library: bool) -> set[str] | None:
    if all_library:
        return None
    selected = batch_file or _latest_top_batch()
    if selected is None:
        print("No top batch file found; falling back to full library scope.")
        return None
    rows = json.loads(selected.read_text(encoding="utf-8"))
    ids = {row["id"] for row in rows if row.get("id")}
    print(f"Batch scope: {selected}")
    print(f"Batch ids: {len(ids)}")
    return ids


def _write_top_song_lyrics(top_songs_dir: Path) -> None:
    for sidecar in sorted(DEFAULT_REVIEW_DIR.glob("*.json")):
        if sidecar.name == "manifest.json":
            continue
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        stem = sidecar.stem
        wav_path = top_songs_dir / f"{stem}.wav"
        if not wav_path.exists():
            continue

        score = payload.get("score", {})
        lyrics = score.get("lyrics") or ""
        title = score.get("title") or stem
        txt_lines = [
            f"Title: {title}",
            f"Score: {score.get('total', '')}",
            f"Category: {payload.get('category', '')}",
            "",
            lyrics.strip(),
            "",
        ]
        (top_songs_dir / f"{stem}.txt").write_text("\n".join(txt_lines), encoding="utf-8")


def run_workflow(
    min_score: int,
    needs_work_score: int,
    limit: int,
    optimize: bool,
    force_optimize: bool,
    clean: bool,
    keep_work_queues: bool,
    batch_file: Path | None,
    all_library: bool,
    max_review: int | None = None,
    use_existing_review: bool = False,
    rvc: bool | None = None,
    humanize: bool = False,
    human_refs_dir: Path = DEFAULT_HUMAN_REFS_DIR,
    rvc_model: str | None = None,
    rvc_index: str | None = None,
    rvc_infer_cmd: str | None = None,
) -> None:
    if use_existing_review:
        review_count = len(list(DEFAULT_REVIEW_DIR.glob("*.wav")))
        print(f"Using existing review queue: {DEFAULT_REVIEW_DIR} ({review_count} WAV files)")
    else:
        track_ids = _load_batch_ids(batch_file, all_library)
        written = package_candidates(
            review_dir=DEFAULT_REVIEW_DIR,
            needs_work_dir=DEFAULT_NEEDS_WORK_DIR,
            instrumental_dir=DEFAULT_INSTRUMENTAL_DIR,
            min_score=min_score,
            needs_work_score=needs_work_score,
            limit=limit,
            clean=clean,
            track_ids=track_ids,
            max_review=max_review,
        )
        print(f"Packaged files: {len(written)}")
        print(f"Review folder: {DEFAULT_REVIEW_DIR}")
        print(f"Needs-work folder: {DEFAULT_NEEDS_WORK_DIR}")
        print(f"Instrumental folder: {DEFAULT_INSTRUMENTAL_DIR}")

    if not optimize:
        return

    if clean:
        _clean_dir(DEFAULT_TOP_SONGS_DIR)

    repair_python = _python_for_vocal_repair()
    if repair_python == sys.executable and not shutil.which("demucs"):
        print("Demucs is not available in the active Python environment; skipping vocal optimization.")
        return

    _run([repair_python, str(VOCAL_REPAIR_DIR / "vocal_repair.py")])
    _run([repair_python, str(VOCAL_REPAIR_DIR / "artifact_segments.py")])
    stem_command = [
        repair_python,
        str(VOCAL_REPAIR_DIR / "stem_pipeline.py"),
        "--input-dir",
        str(DEFAULT_REVIEW_DIR),
        "--optimized-dir",
        str(DEFAULT_TOP_SONGS_DIR),
        "--profile",
        "transparent",
        "--noise-profile",
        "transparent",
        "--vocal-intro-mute-seconds",
        "8",
        "--vocal-artifact-max-score",
        "22",
    ]
    if humanize:
        stem_command.extend(["--humanize", "--human-refs-dir", str(human_refs_dir)])
        print(f"Humanize: enabled with refs {human_refs_dir}")
    else:
        print("Humanize: skipped")
    if force_optimize:
        stem_command.append("--force")
    if rvc is True:
        stem_command.append("--rvc")
        print("RVC: enabled explicitly")
    else:
        stem_command.append("--no-rvc")
        print("RVC: skipped by default")
    if rvc_model:
        stem_command.extend(["--rvc-model", rvc_model])
    if rvc_index:
        stem_command.extend(["--rvc-index", rvc_index])
    if rvc_infer_cmd:
        stem_command.extend(["--rvc-infer-cmd", rvc_infer_cmd])
    _run(stem_command)
    _write_top_song_lyrics(DEFAULT_TOP_SONGS_DIR)
    print(f"Top songs folder: {DEFAULT_TOP_SONGS_DIR}")

    if not keep_work_queues:
        _remove_work_queue(DEFAULT_REVIEW_DIR)
        _remove_work_queue(DEFAULT_INSTRUMENTAL_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Package top songs and run vocal-stem optimization.")
    parser.add_argument("--min-score", type=int, default=85, help="Score required for the top-song review queue")
    parser.add_argument("--needs-work-score", type=int, default=60, help="Score required for needs-work tracking")
    parser.add_argument("--limit", type=int, default=200, help="Max succeeded jobs to inspect")
    parser.add_argument("--skip-optimize", action="store_true", help="Only package queues; do not run vocal repair")
    parser.add_argument("--force-optimize", action="store_true", help="Re-run Demucs even when remix.wav exists")
    parser.add_argument("--no-clean", action="store_true", help="Keep existing queue files")
    parser.add_argument("--keep-work-queues", action="store_true", help="Keep review/instrumental intermediate queues")
    parser.add_argument("--batch-file", type=Path, default=None, help="Only process songs listed in this batch JSON")
    parser.add_argument("--all-library", action="store_true", help="Process the whole library instead of latest top batch")
    parser.add_argument("--max-review", type=int, default=20, help="Only optimize the top N passing songs")
    parser.add_argument("--use-existing-review", action="store_true", help="Skip scoring/package and optimize the current review queue")
    parser.add_argument("--rvc", action="store_true", help="Require RVC success (off by default)")
    parser.add_argument("--humanize", action="store_true", help="Apply experimental human-reference vocal humanization")
    parser.add_argument("--no-humanize", action="store_true", help="Accepted for compatibility; humanize is off by default")
    parser.add_argument("--human-refs-dir", type=Path, default=DEFAULT_HUMAN_REFS_DIR)
    parser.add_argument("--no-rvc", action="store_true", help="Accepted for compatibility; RVC is already off by default")
    parser.add_argument("--rvc-model", default=None, help="RVC model path")
    parser.add_argument("--rvc-index", default=None, help="RVC index path")
    parser.add_argument("--rvc-infer-cmd", default=None, help="RVC command template with {input} {output} {model} {index}")
    args = parser.parse_args()

    rvc_flag: bool | None
    if args.no_rvc:
        rvc_flag = False
    elif args.rvc:
        rvc_flag = True
    else:
        rvc_flag = False

    run_workflow(
        min_score=args.min_score,
        needs_work_score=args.needs_work_score,
        limit=args.limit,
        optimize=not args.skip_optimize,
        force_optimize=args.force_optimize,
        clean=not args.no_clean,
        keep_work_queues=args.keep_work_queues,
        batch_file=args.batch_file,
        all_library=args.all_library,
        max_review=args.max_review,
        use_existing_review=args.use_existing_review,
        rvc=rvc_flag,
        humanize=args.humanize and not args.no_humanize,
        human_refs_dir=args.human_refs_dir,
        rvc_model=args.rvc_model,
        rvc_index=args.rvc_index,
        rvc_infer_cmd=args.rvc_infer_cmd,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
