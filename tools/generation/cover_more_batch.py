"""Generate full-song fixed-voice SVC covers for a list of source songs.

Serial by design: one SoulX-SVC job at a time. Before each song it waits for
enough free VRAM (another session may be holding the GPU for image/video work),
and it never starts a song after the configured stop time. Every state change is
written to a JSON status file so a detached run can be inspected or resumed.

Usage:
  python tools/generation/cover_more_batch.py \
      --list inputs/workflows/2026-10-20_more_covers.json \
      --status outputs/2026-10-20/_work/more_covers/status.json \
      --stop-at 16:30 --min-free-mib 6000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "data/voice_studio/jobs"
PY = ROOT / "tools/vocal_repair/.venv/Scripts/python.exe"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def gpu_free_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def job_ready(slug: str) -> bool:
    manifest = JOBS / f"cover_{slug}_full_v1" / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    outputs = data.get("outputs", {})
    return (data.get("status") in {"awaiting_listening", "technical_review_required"}
            and bool(outputs.get("mix")) and bool(outputs.get("vocals")))


def run_cover(entry: dict, log_path: Path) -> int:
    source = (ROOT / entry["source"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    command = [
        "cmd", "/c", str(ROOT / "run_workflows.bat"), "voice", "cover",
        "--workspace", str(ROOT / "data/voice_studio"),
        "--voice-id", "iron_rain_female_v1",
        "--source", str(source),
        "--seconds", "0",
        "--job-id", f"cover_{entry['slug']}_full_v1",
        "--seed", "42",
        "--steps", "32",
        "--execute",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace")
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--stop-at", default="16:30", help="Do not start a new song after this local time")
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--only", nargs="*", help="Optional slug subset")
    args = parser.parse_args()

    entries = json.loads(args.list.read_text(encoding="utf-8"))
    if args.only:
        entries = [item for item in entries if item["slug"] in set(args.only)]
    state = {"started_at": now(), "stop_at": args.stop_at, "songs": {}}
    if args.status.is_file():
        try:
            state = json.loads(args.status.read_text(encoding="utf-8"))
        except ValueError:
            pass
    state.setdefault("songs", {})
    state.update(stop_at=args.stop_at, updated_at=now())
    write_json(args.status, state)

    log_dir = args.status.parent / "logs"
    for entry in entries:
        slug = entry["slug"]
        if job_ready(slug):
            state["songs"].setdefault(slug, {})["status"] = "ready"
            state["songs"][slug]["checked_at"] = now()
            write_json(args.status, state)
            log(f"SKIP {slug}: completed job already present")
            continue
        if datetime.now().strftime("%H:%M") >= args.stop_at:
            state["songs"].setdefault(slug, {})["status"] = "not_started_stop_time"
            write_json(args.status, state)
            log(f"STOP {slug}: past the configured stop time")
            continue

        # Wait for the GPU without ever killing another session's process.
        while True:
            free = gpu_free_mib()
            if free is None or free >= args.min_free_mib:
                break
            if datetime.now().strftime("%H:%M") >= args.stop_at:
                break
            log(f"WAIT {slug}: {free} MiB free, need {args.min_free_mib}")
            time.sleep(30)
        if datetime.now().strftime("%H:%M") >= args.stop_at:
            state["songs"].setdefault(slug, {})["status"] = "not_started_stop_time"
            write_json(args.status, state)
            log(f"STOP {slug}: past the configured stop time while waiting for the GPU")
            continue

        state["songs"].setdefault(slug, {}).update(
            status="running", started_at=now(), source=entry["source"],
            title_zh=entry.get("title_zh"), origin=entry.get("origin"))
        write_json(args.status, state)
        log(f"START {slug} (free VRAM {gpu_free_mib()} MiB)")
        code = run_cover(entry, log_dir / f"{slug}.log")
        ready = job_ready(slug)
        state["songs"][slug].update(
            status="ready" if ready and code == 0 else "failed",
            exit_code=code, finished_at=now())
        write_json(args.status, state)
        log(f"{'DONE' if ready and code == 0 else 'FAILED'} {slug} exit={code}")

    state["finished_at"] = now()
    state["updated_at"] = now()
    write_json(args.status, state)
    log("batch finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
