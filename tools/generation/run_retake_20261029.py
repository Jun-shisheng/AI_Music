"""Retakes for the two titles the 10-29 review flagged.

- 02_third_bell: 224 s, outside the 215 s window.  Same tags and lyrics,
  `--max-plan-seconds 200` so the accepted plan has to be shorter.
- 04_read_the_roll: climax at 56% (early).  Same tags, new seed; only swap it
  in if the new take is still build-up with the peak later and coverage >=0.75.

Both render into `outputs/2026-10-29_retake/` so the delivered files are not
touched until the numbers are compared.

Usage:
  vendor/YuE2/.venv/Scripts/python.exe -I tools/generation/run_retake_20261029.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "inputs" / "2026-10-29_film_ref"
OUT = ROOT / "outputs" / "2026-10-29_retake"
LOGS = ROOT / "logs"
PY = ROOT / "vendor" / "YuE2" / ".venv" / "Scripts" / "python.exe"
BATCH = ROOT / "tools" / "generation" / "yue2_batch_battle.py"
QA_PY = ROOT / "tools" / "vocal_repair" / ".venv" / "Scripts" / "python.exe"
STYLE_QA = ROOT / "tools" / "generation" / "style_qa.py"

TAKES = [
    {"slug": "02_third_bell", "mode": "epic", "seed": 9950,
     "max_plan": 200.0},
    {"slug": "04_read_the_roll", "mode": "epic_ethereal", "seed": 9960,
     "max_plan": 215.0},
]


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = LOGS / "retake_20261029_status.json"
    status: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "titles": {}}

    for take in TAKES:
        slug, mode, seed_base = take["slug"], take["mode"], take["seed"]
        selected = OUT / "selected" / f"{slug}.wav"
        if selected.exists():
            print(f"[{time.strftime('%H:%M:%S')}] {slug}: already done", flush=True)
            continue
        lift = "0" if mode in ("epic", "epic_ethereal") else "2"
        intro = 8 if mode in ("epic", "epic_ethereal") else 4
        cmd = [
            str(PY), "-I", str(BATCH),
            "--inputs", str(INPUTS), "--out-dir", str(OUT),
            "--only", slug, "--candidates", "1", "--retries", "1",
            "--seed-base", str(seed_base), "--budget", "12",
            "--nar-query-chunk", "1024", "--gpu-mem-fraction", "0.90",
            "--max-plan-seconds", str(take["max_plan"]),
            "--intro-bars", str(intro),
            "--melody-target", "diagnostic", "--hook-max-tile", "3.5",
            "--no-early-accept", "--shape", "hook chorus",
            "--shape", "thin verse:2", "--final-lift", lift,
        ]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOGS / f"retake_{slug}_{stamp}.log"
        status["titles"][slug] = {"state": "running", "log": str(log_path),
                                  "command": cmd}
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {slug} (seed {seed_base})", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace")
        entry = {"returncode": proc.returncode,
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        if proc.returncode == 0 and selected.exists():
            entry["state"] = "done"
            subprocess.run([str(QA_PY), "-I", str(STYLE_QA),
                            "--file", str(selected),
                            "--json", str(OUT / "qa" / f"{slug}_style.json")],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
            result = next(iter(OUT.glob(f"candidates/{slug}_*/result.json")), None)
            if result:
                try:
                    data = json.loads(result.read_text(encoding="utf-8"))
                    entry.update({"duration_s": data["arc"]["duration_s"],
                                  "arc": data["arc"]["verdict"],
                                  "lift_db": data["arc"]["lift_db"],
                                  "peak_at": data["arc"]["peak_at"],
                                  "coverage": data.get("diction")})
                except Exception:
                    pass
        else:
            entry["state"] = "failed"
        status["titles"][slug].update(entry)
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {slug}: {entry['state']} "
              f"{entry.get('duration_s','')}s {entry.get('arc','')} "
              f"peak={entry.get('peak_at')} cov={entry.get('coverage')}",
              flush=True)

    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
