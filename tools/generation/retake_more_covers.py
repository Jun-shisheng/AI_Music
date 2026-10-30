"""Retake a few weak fixed-voice covers with a different seed / step count.

Runs one SoulX-SVC job at a time under new job ids (`cover_<slug>_full_v2`),
never touching the full_v1 jobs. The caller decides afterwards whether a v2
take beats v1 on the content proxy; this script only produces the takes.
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


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def gpu_free_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=False)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def ready(slug: str) -> bool:
    path = JOBS / f"cover_{slug}_full_v2" / "manifest.json"
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    outputs = data.get("outputs", {})
    return (data.get("status") in {"awaiting_listening", "technical_review_required"}
            and bool(outputs.get("mix")) and bool(outputs.get("vocals")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--slugs", nargs="+", required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--stop-at", default="16:20")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--steps", type=int, default=48)
    args = ap.parse_args()
    entries = {e["slug"]: e for e in json.loads(args.list.read_text(encoding="utf-8"))}
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for slug in args.slugs:
        if datetime.now().strftime("%H:%M") >= args.stop_at:
            print(f"[{now()}] STOP before {slug}: past stop time", flush=True)
            break
        if ready(slug):
            print(f"[{now()}] SKIP {slug}: v2 already complete", flush=True)
            continue
        while (gpu_free_mib() or 0) < 6000:
            if datetime.now().strftime("%H:%M") >= args.stop_at:
                break
            time.sleep(30)
        entry = entries[slug]
        source = (ROOT / entry["source"]).resolve()
        command = ["cmd", "/c", str(ROOT / "run_workflows.bat"), "voice", "cover",
                   "--workspace", str(ROOT / "data/voice_studio"),
                   "--voice-id", "iron_rain_female_v1",
                   "--source", str(source), "--seconds", "0",
                   "--job-id", f"cover_{slug}_full_v2",
                   "--seed", str(args.seed), "--steps", str(args.steps), "--execute"]
        print(f"[{now()}] START {slug} v2 seed={args.seed} steps={args.steps}", flush=True)
        with (args.log_dir / f"{slug}_v2.log").open("w", encoding="utf-8") as handle:
            code = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                    stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace").wait()
        print(f"[{now()}] {'DONE' if ready(slug) and code == 0 else 'FAILED'} {slug} exit={code}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
