"""Sequential full-song fixed-voice covers for the 2026-10-16 hook batch.

Runs one SVC job per title through the unified workflow entry, so the workflow
lock and run log are reused. Existing completed jobs are skipped, which makes
the driver safe to restart. No audio post-processing is applied.

Defaults reproduce the first pass (`full_v1`, seed 42, 32 steps). Pass
`--variant full_v2 --seed 43 --steps 48` to make an alternative take for A/B.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / "tools/vocal_repair/.venv/Scripts/python.exe"
SOURCE_DIR = ROOT / "_archive/outputs_history/2026-10-16_epic_hook/selected"
JOBS = ROOT / "data/voice_studio/jobs"
DEFAULT_SLUGS = ["01_hammer_and_hymn", "03_the_gate_holds", "04_iron_rain",
                 "05_drums_below_deck", "06_count_the_living"]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def finished(job_id: str) -> bool:
    manifest = JOBS / job_id / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("status") in {"awaiting_listening", "technical_review_required"} and bool(
        data.get("outputs", {}).get("mix"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slugs", nargs="+", default=DEFAULT_SLUGS)
    parser.add_argument("--variant", default="full_v1")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--chunk-seconds", type=float)
    args = parser.parse_args()

    failures: list[str] = []
    for slug in args.slugs:
        source = SOURCE_DIR / f"{slug}.wav"
        job_id = f"cover_{slug}_{args.variant}"
        if not source.is_file():
            print(f"{now()} MISSING source {source}", flush=True)
            failures.append(job_id)
            continue
        if finished(job_id):
            print(f"{now()} SKIP {job_id} (already complete)", flush=True)
            continue
        command = [
            str(PY), str(ROOT / "tools/music_workflows.py"), "voice", "cover",
            "--workspace", str(ROOT / "data/voice_studio"),
            "--voice-id", "iron_rain_female_v1",
            "--source", str(source),
            "--seconds", "0",
            "--job-id", job_id,
        ]
        if args.seed is not None:
            command += ["--seed", str(args.seed)]
        if args.steps is not None:
            command += ["--steps", str(args.steps)]
        if args.chunk_seconds is not None:
            command += ["--chunk-seconds", str(args.chunk_seconds)]
        command.append("--execute")
        print(f"{now()} START {job_id}", flush=True)
        code = subprocess.run(command, cwd=ROOT, env=dict(
            os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")).returncode
        print(f"{now()} END {job_id} exit={code}", flush=True)
        if code != 0 or not finished(job_id):
            failures.append(job_id)
    print(f"{now()} BATCH DONE failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
