"""Main epic batch for the 2026-10-28 evening review.

Round 1/2 calibration chose the direction: 御姐 / 哲思史诗 (build-up + clear
diction) with the fixed ethereal and battle tag stacks.  This driver renders
every title already written into `inputs/2026-10-28_epic_batch/`.

One process per title: the CUDA caching allocator fragments across titles on
Windows and `expandable_segments` is not supported there.

Usage:
  vendor/YuE2/.venv/Scripts/python.exe -I tools/generation/run_epic_batch_20261028.py
  ... --only 01,07        # resume / drill into specific titles
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "inputs" / "2026-10-28_epic_batch"
OUT = ROOT / "outputs" / "2026-10-28_epic_batch"
LOGS = ROOT / "logs"
PY = ROOT / "vendor" / "YuE2" / ".venv" / "Scripts" / "python.exe"
BATCH = ROOT / "tools" / "generation" / "yue2_batch_battle.py"
STYLE_QA = ROOT / "tools" / "generation" / "style_qa.py"
QA_PY = ROOT / "tools" / "vocal_repair" / ".venv" / "Scripts" / "python.exe"


def mode_shapes(mode: str) -> list[str]:
    return ["--shape", "hook chorus", "--shape", "thin verse:2",
            "--final-lift", "2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--candidates", type=int, default=1)
    ap.add_argument("--max-plan-seconds", type=float, default=200.0)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--intro-bars", type=int, default=None,
                    help="override the per-mode intro trim")
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()

    modes = json.loads((INPUTS / "_modes.json").read_text(encoding="utf-8"))
    LOGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = LOGS / "epic_batch_20261028_status.json"
    status: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "titles": {}}

    slugs = sorted(modes)
    if args.only:
        needles = [n.strip() for n in args.only.split(",") if n.strip()]
        slugs = [s for s in slugs if any(n in s for n in needles)]

    for index, slug in enumerate(slugs):
        mode = modes[slug]
        selected = OUT / "selected" / f"{slug}.wav"
        if selected.exists():
            status["titles"][slug] = {"state": "skipped"}
            continue
        seed = 8100 + index * 100 + args.seed_offset
        cmd = [
            str(PY), "-I", str(BATCH),
            "--inputs", str(INPUTS), "--out-dir", str(OUT),
            "--only", slug, "--candidates", str(args.candidates),
            "--retries", str(args.retries), "--seed-base", str(seed),
            "--budget", "12", "--nar-query-chunk", "1024",
            "--gpu-mem-fraction", "0.90",
            "--max-plan-seconds", str(args.max_plan_seconds),
            "--intro-bars", str(args.intro_bars if args.intro_bars is not None
                                  else (6 if mode == "epic_ethereal" else 4)),
            "--melody-target", "diagnostic", "--hook-max-tile", "3.5",
            "--no-early-accept", *mode_shapes(mode),
        ]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOGS / f"epicb_{slug}_{stamp}.log"
        status["titles"][slug] = {"state": "running", "mode": mode,
                                  "started": stamp, "log": str(log_path),
                                  "command": cmd}
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {slug} ({mode})", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace")
        ok = proc.returncode == 0 and selected.exists()
        entry = {"state": "done" if ok else "failed",
                 "returncode": proc.returncode,
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        if ok:
            qa = subprocess.run(
                [str(QA_PY), "-I", str(STYLE_QA), "--file", str(selected),
                 "--json", str(OUT / "qa" / f"{slug}_style.json")],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            entry["style_qa"] = qa.stdout.strip().splitlines()[-3:]
            result = next(iter(OUT.glob(f"candidates/{slug}_*/result.json")), None)
            if result:
                try:
                    data = json.loads(result.read_text(encoding="utf-8"))
                    entry["duration_s"] = data["arc"]["duration_s"]
                    entry["arc"] = data["arc"]["verdict"]
                    entry["lift_db"] = data["arc"]["lift_db"]
                    entry["peak_at"] = data["arc"]["peak_at"]
                    entry["coverage"] = data.get("diction")
                except Exception:
                    pass
        status["titles"][slug].update(entry)
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {slug}: "
              f"{'done' if ok else 'FAILED rc=' + str(proc.returncode)} "
              f"{entry.get('arc', '')} {entry.get('duration_s', '')}s "
              f"cov={entry.get('coverage')}", flush=True)

    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
