"""Does the native stereo width depend on the seed enough to select for it?

The 10-29 batches measured 0.212-0.372 across eight titles, against 0.50 for
the film references.  Tags and the TSFH NAR LoRA both failed to move it, so
the last native lever is take selection: render the same title from two more
seed bases and measure.  If the spread within one title is small (~0.02), no
amount of retaking will reach 0.45; if it is large (>0.08), picking the wide
take is a legitimate, unprocessed improvement.

Title: 07_ash_map (the narrowest of the batch, 0.212).
Seeds: 9700 and 9800 (the per-candidate seeds are seed_base + attempt*101 + i*37).

Usage:
  vendor/YuE2/.venv/Scripts/python.exe -I tools/generation/run_width_seeds_20261029.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "inputs" / "2026-10-29_film_ref2"
OUT = ROOT / "outputs" / "2026-10-29_width_seeds"
LOGS = ROOT / "logs"
PY = ROOT / "vendor" / "YuE2" / ".venv" / "Scripts" / "python.exe"
BATCH = ROOT / "tools" / "generation" / "yue2_batch_battle.py"
QA_PY = ROOT / "tools" / "vocal_repair" / ".venv" / "Scripts" / "python.exe"
STYLE_QA = ROOT / "tools" / "generation" / "style_qa.py"

SLUG = "07_ash_map"
SEEDS = [9700, 9800]


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = LOGS / "width_seeds_20261029_status.json"
    status: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "takes": {}}

    for seed in SEEDS:
        out_dir = OUT / f"s{seed}"
        selected = out_dir / "selected" / f"{SLUG}.wav"
        if selected.exists():
            print(f"[{time.strftime('%H:%M:%S')}] s{seed}: already done", flush=True)
            continue
        cmd = [
            str(PY), "-I", str(BATCH),
            "--inputs", str(INPUTS), "--out-dir", str(out_dir),
            "--only", SLUG, "--candidates", "1", "--retries", "1",
            "--seed-base", str(seed), "--budget", "12",
            "--nar-query-chunk", "1024", "--gpu-mem-fraction", "0.90",
            "--max-plan-seconds", "215", "--intro-bars", "8",
            "--melody-target", "diagnostic", "--hook-max-tile", "3.5",
            "--no-early-accept", "--shape", "hook chorus",
            "--shape", "thin verse:2", "--final-lift", "0",
        ]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOGS / f"width_s{seed}_{stamp}.log"
        status["takes"][f"s{seed}"] = {"state": "running", "log": str(log_path),
                                        "command": cmd}
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] seed {seed}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace")
        entry = {"returncode": proc.returncode,
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        if proc.returncode == 0 and selected.exists():
            entry["state"] = "done"
            subprocess.run([str(QA_PY), "-I", str(STYLE_QA),
                            "--file", str(selected),
                            "--json", str(out_dir / "style.json")],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
            result = next(iter(out_dir.glob(f"candidates/{SLUG}_*/result.json")), None)
            if result:
                try:
                    data = json.loads(result.read_text(encoding="utf-8"))
                    entry.update({"duration_s": data["arc"]["duration_s"],
                                  "arc": data["arc"]["verdict"],
                                  "coverage": data.get("diction")})
                except Exception:
                    pass
        else:
            entry["state"] = "failed"
        status["takes"][f"s{seed}"].update(entry)
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] s{seed}: {entry['state']}", flush=True)

    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
