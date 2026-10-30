"""Second 2026-10-29 film batch, rendered with the NAR-LoRA winner.

The first batch (5 titles) fixed the arrangement palette but not the stereo
width.  `run_lora_ab_20261029.py` renders the same title with NAR LoRA
scales 0 / 0.15 / 0.25; this driver takes the winning scale and renders the
three new titles in `inputs/2026-10-29_film_ref2/`.

Usage:
  vendor/YuE2/.venv/Scripts/python.exe -I tools/generation/run_film2_20261029.py \
      --lora-nar models/yue2/loras/tsfh-steps-from-hell/adapter-nar-194/lora.safetensors \
      --nar-scale 0.15
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "inputs" / "2026-10-29_film_ref2"
OUT = ROOT / "outputs" / "2026-10-29_film_ref2"
LOGS = ROOT / "logs"
PY = ROOT / "vendor" / "YuE2" / ".venv" / "Scripts" / "python.exe"
BATCH = ROOT / "tools" / "generation" / "yue2_batch_battle.py"
STYLE_QA = ROOT / "tools" / "generation" / "style_qa.py"
QA_PY = ROOT / "tools" / "vocal_repair" / ".venv" / "Scripts" / "python.exe"

SLOW_MODES = {"epic", "epic_ethereal"}


def mode_shapes(mode: str) -> list[str]:
    lift = "0" if mode in SLOW_MODES else "2"
    return ["--shape", "hook chorus", "--shape", "thin verse:2",
            "--final-lift", lift]


def intro_bars(mode: str) -> int:
    return 8 if mode in SLOW_MODES else 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--candidates", type=int, default=1)
    ap.add_argument("--max-plan-seconds", type=float, default=215.0)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--intro-bars", type=int, default=None)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--lora-nar", type=Path, default=None)
    ap.add_argument("--nar-scale", type=float, default=1.0)
    args = ap.parse_args()

    modes = json.loads((INPUTS / "_modes.json").read_text(encoding="utf-8"))
    LOGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = LOGS / "film_ref2_20261029_status.json"
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
        seed = 9400 + index * 100 + args.seed_offset
        cmd = [
            str(PY), "-I", str(BATCH),
            "--inputs", str(INPUTS), "--out-dir", str(OUT),
            "--only", slug, "--candidates", str(args.candidates),
            "--retries", str(args.retries), "--seed-base", str(seed),
            "--budget", "12", "--nar-query-chunk", "1024",
            "--gpu-mem-fraction", "0.90",
            "--max-plan-seconds", str(args.max_plan_seconds),
            "--intro-bars", str(args.intro_bars if args.intro_bars is not None
                                  else intro_bars(mode)),
            "--melody-target", "diagnostic", "--hook-max-tile", "3.5",
            "--no-early-accept", *mode_shapes(mode),
        ]
        if args.lora_nar is not None:
            cmd += ["--lora-nar", str(args.lora_nar),
                    "--nar-scale", str(args.nar_scale)]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOGS / f"film2_{slug}_{stamp}.log"
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
