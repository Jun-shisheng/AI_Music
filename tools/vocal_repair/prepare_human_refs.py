from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "human_refs" / "raw"
DEFAULT_VOCALS_DIR = PROJECT_ROOT / "human_refs" / "vocals"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


def _python() -> str:
    venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _safe_name(path: Path) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in path.stem).strip("._") or "ref"


def iter_audio_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)


def duration_seconds(path: Path) -> float:
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            stderr=subprocess.DEVNULL,
        )
        payload = json.loads(output)
        return float(payload.get("format", {}).get("duration") or 0)
    except Exception:
        return 0.0


def demucs_output_for(work_dir: Path, source: Path) -> Path:
    return work_dir / "htdemucs" / source.stem / "vocals.wav"


def separate_reference(source: Path, raw_dir: Path, vocals_dir: Path, force: bool = False) -> Path:
    relative_parent = source.parent.relative_to(raw_dir)
    target_dir = vocals_dir / relative_parent
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{_safe_name(source)}_vocals.wav"
    if target.exists() and not force:
        return target

    work_dir = target_dir / f".demucs_{_safe_name(source)[:32]}"
    subprocess.run(
        [
            _python(),
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "-n",
            "htdemucs",
            "-o",
            str(work_dir),
            str(source),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    vocals = demucs_output_for(work_dir, source)
    if not vocals.exists():
        raise FileNotFoundError(f"Demucs did not write vocals stem: {vocals}")
    shutil.copy2(vocals, target)
    shutil.rmtree(work_dir, ignore_errors=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Separate collected human vocal references from accompanied audio.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--vocals-dir", type=Path, default=DEFAULT_VOCALS_DIR)
    parser.add_argument("--force", action="store_true", help="Re-separate references that already have vocal stems")
    parser.add_argument("--min-duration", type=float, default=30, help="Skip files shorter than this many seconds")
    parser.add_argument("--max-duration", type=float, default=420, help="Skip files longer than this many seconds")
    parser.add_argument("--max-files", type=int, default=12, help="Maximum number of references to separate")
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.vocals_dir.mkdir(parents=True, exist_ok=True)
    sources = iter_audio_files(args.raw_dir)
    if not sources:
        print(f"No audio files found in {args.raw_dir}")
        return 0

    selected: list[Path] = []
    skipped: list[tuple[Path, float, str]] = []
    for source in sources:
        duration = duration_seconds(source)
        if duration < args.min_duration:
            skipped.append((source, duration, "too_short"))
            continue
        if duration > args.max_duration:
            skipped.append((source, duration, "too_long"))
            continue
        selected.append(source)
        if len(selected) >= args.max_files:
            break

    for source, duration, reason in skipped:
        print(f"skip {reason}: {duration:.1f}s {source}")

    for index, source in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {duration_seconds(source):.1f}s {source}")
        target = separate_reference(source, args.raw_dir, args.vocals_dir, force=args.force)
        print(f"  vocals -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
