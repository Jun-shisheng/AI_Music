"""HeartMuLa generation backend for the FastAPI service (new source-clean path).

Runs the redesigned pipeline: lyrics + tags -> HeartMuLa (official recommended
weights) -> native 48 kHz WAV -> linear loudness normalization. NO post
processing (no frame repair / EQ / humanize / de-esser).

8GB GPU config (measured): cfg_scale=1.0 (single batch, ~3.4 it/s) with the
codec on CPU. Candidate diversity is not used here — a single take is generated
per job, matching the API's job model; use tools/generation/clean_pipeline.py
directly for multi-candidate listening selection.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

from app.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ACE_PYTHON = PROJECT_ROOT / "vendor" / "ACE-Step-1.5" / ".venv" / "Scripts" / "python.exe"
PIPELINE_SCRIPT = PROJECT_ROOT / "tools" / "generation" / "clean_pipeline.py"


def _run_blocking_sync(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=3600)
    except subprocess.TimeoutExpired:
        return 1, "generation timed out after 3600s"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


async def _run_blocking(cmd: list[str]) -> tuple[int, str]:
    """Run the generator subprocess off the event loop.

    asyncio's own subprocess support needs a Proactor event loop on Windows,
    which uvicorn does not install (under --reload it runs a Selector loop and
    create_subprocess_exec raises NotImplementedError). A worker thread has no
    such platform restriction.
    """
    return await asyncio.to_thread(_run_blocking_sync, cmd)


def _tags_from_prompt(style_prompt: str) -> str:
    """Map the UI's style_prompt (comma list) to HeartMuLa tags."""
    parts = [p.strip() for p in style_prompt.split(",") if p.strip()]
    # drop UI-only tokens that HeartMuLa does not understand well
    skip = {"bpm"}
    keep = []
    for part in parts:
        if any(token in part.lower() for token in skip):
            continue
        keep.append(part)
    return ", ".join(keep) if keep else "pop, female vocal, clean production"


def _lyrics_text(lyrics: str, instrumental: bool) -> str:
    if instrumental:
        return "[Instrumental]"
    return lyrics


async def generate(
    lyrics: str,
    style_prompt: str,
    output_path: str,
    instrumental: bool = False,
    duration_seconds: int | None = None,
    loopable: bool = False,
) -> Path:
    """Generate a song with HeartMuLa through the clean pipeline.

    Returns the output path. Raises on failure so the job is marked failed.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    seconds = duration_seconds or 30
    # HeartMuLa max practical single pass on 8GB; loopable is not specially
    # supported by this backend yet.
    lyrics_txt = out.parent / f"{out.stem}_lyrics.txt"
    tags_txt = out.parent / f"{out.stem}_tags.txt"
    lyrics_txt.write_text(_lyrics_text(lyrics, instrumental), encoding="utf-8")
    tags_txt.write_text(_tags_from_prompt(style_prompt), encoding="utf-8")

    cmd = [
        str(ACE_PYTHON), str(PIPELINE_SCRIPT),
        "--lyrics", str(lyrics_txt), "--tags", str(tags_txt),
        "--title", out.stem, "--candidates", "1", "--seconds", str(seconds),
        # clean_pipeline writes to <out-dir>/<title>_<stamp>/, so the out-dir must
        # be the "candidates" folder the lookup below globs.
        "--out-dir", str(out.parent / "candidates"),
    ]
    # Long songs need the split token-gen/codec-decode path (stable on 8GB).
    if seconds >= 60:
        cmd.append("--full-song")
    returncode, log = await _run_blocking(cmd)
    if returncode != 0:
        raise RuntimeError(f"HeartMuLa generation failed ({returncode}): {log[-1500:]}")

    # clean_pipeline writes to outputs/candidates/<title>_<stamp>/<title>_c1.wav
    # copy the take to the job's output path
    stamp_dirs = sorted((out.parent / "candidates").glob(f"{out.stem}_*"), reverse=True)
    if not stamp_dirs:
        raise RuntimeError(f"HeartMuLa output not found for {out.stem}")
    take = stamp_dirs[0] / f"{out.stem}_c1.wav"
    if not take.exists():
        raise RuntimeError(f"HeartMuLa take missing: {take}")
    shutil.copy2(take, out)
    return out
