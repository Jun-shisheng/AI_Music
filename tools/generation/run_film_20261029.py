"""2026-10-29 batch: film-score references (《凡人修仙传》/《灵笼》).

The 10-28 batch measured narrower (0.23-0.46) and lighter in the sub
(27-31 dB) than the three references the user named (0.50 width, 32 dB sub,
12-21 dB dynamic range).  This driver keeps the 10-28 render pipeline but
changes what the tags and the plan ask for:

- eight-bar intros (four for the battle title) so the first sung line lands
  after the scene is set, not at 4 s;
- no `final-lift` on the two slow palettes.  ref_profile measured the named
  references at -3 to -7 semitones from verse to chorus: the orchestra rises,
  the voice stays in the chest.  The 10-28 batch transposed the last chorus
  +2 semitones, which is the pop move; the battle title keeps it.
- `thin verse:2` stays: density swing in the references is x1.54-x1.68 and
  the 10-28 batch was already x1.6-2.3, so the verse/chorus word-rate
  contrast was never the problem.

One process per title: the CUDA caching allocator fragments across titles on
Windows and `expandable_segments` is not supported there.

Usage:
  vendor/YuE2/.venv/Scripts/python.exe -I tools/generation/run_film_20261029.py
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
INPUTS = ROOT / "inputs" / "2026-10-29_film_ref"
OUT = ROOT / "outputs" / "2026-10-29_film_ref"
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
    ap.add_argument("--intro-bars", type=int, default=None,
                    help="override the per-mode intro trim")
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()

    modes = json.loads((INPUTS / "_modes.json").read_text(encoding="utf-8"))
    LOGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = LOGS / "film_ref_20261029_status.json"
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
        seed = 9100 + index * 100 + args.seed_offset
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
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOGS / f"film_{slug}_{stamp}.log"
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
