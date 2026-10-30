"""Sequential YuE2 driver for the 2026-10-26 Chinese lyric/melody pilot.

One process per title on purpose: the 2026-10-13 and 2026-10-22 runs showed
that the CUDA caching allocator fragments across titles inside one process, and
`expandable_segments` is not supported on Windows.  A fresh process per title
also makes a failed title cheap to retry without touching the successful ones.

Per-mode plan shaping:
  narrative / guofeng : hook chorus + thin verse:2 + split chorus:6
  atmospheric / epic  : hook chorus + thin verse:2 (no split; the song is
                        supposed to breathe and let long notes ring)
  all                 : final chorus +2 semitones

The lyrics and tags come from `inputs/2026-10-26_zh_pop/`; this driver does not
write lyrics, it only renders them.

Usage:
  vendor/YuE2/.venv/Scripts/python.exe -I tools/generation/run_zh_pop_20261026.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "inputs" / "2026-10-26_zh_pop"
OUT = ROOT / "outputs" / "2026-10-26_zh_pop"
LOGS = ROOT / "logs"
PY = ROOT / "vendor" / "YuE2" / ".venv" / "Scripts" / "python.exe"
BATCH = ROOT / "tools" / "generation" / "yue2_batch_battle.py"

SEED_BASE = {
    "01_last_bus_convenience_store": 5100,
    "02_umbrella_on_the_wrong_side": 5200,
    "03_elevator_at_three": 5300,
    "04_before_the_gate_opens": 5400,
    "05_last_load_laundromat": 5500,
    "06_lantern_on_the_pass": 5600,
    "07_ride_to_the_last_stop": 5700,
    "08_give_the_name_back_to_the_sea": 5800,
    "09_time_is_a_circle": 5900,
    "10_borrowed_light": 6000,
    "11_banner_in_the_ash": 6100,
    "12_names_not_finished": 6200,
}

MODES = json.loads((INPUTS / "_modes.json").read_text(encoding="utf-8"))
SLUGS = list(MODES)


def shape_args(mode: str) -> list[str]:
    args = ["--shape", "hook chorus"]
    if mode in ("atmospheric", "epic", "epic_rap"):
        # Epic / BGM wants sustained notes and space, not a machine-gun
        # chorus: thin the verse, keep the chorus's long notes, let the
        # cinematic arrangement carry the size.
        args += ["--shape", "thin verse:2"]
    else:
        # thin verse:2 (not 1.5) is what actually creates the contrast: 1.5
        # was measured to leave the verse density unchanged on real plans.
        # split 6 (not 4) lifts the chorus without turning every long note into
        # a same-pitch repeat.
        args += ["--shape", "thin verse:2", "--shape", "split chorus:6"]
    args += ["--final-lift", "2"]
    return args


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma-separated slug substrings to run")
    ap.add_argument("--candidates", type=int, default=2,
                    help="takes per title (default 2)")
    args = ap.parse_args()
    slugs = SLUGS
    if args.only:
        needles = [n.strip() for n in args.only.split(",") if n.strip()]
        slugs = [s for s in SLUGS if any(n in s for n in needles)]
    LOGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = LOGS / "zh_pop_20261026_status.json"
    status: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "titles": {}}

    for slug in slugs:
        mode = MODES[slug]
        selected = OUT / "selected" / f"{slug}.wav"
        if selected.exists():
            status["titles"][slug] = {"state": "skipped", "reason": "already promoted"}
            continue
        cmd = [
            str(PY), "-I", str(BATCH),
            "--inputs", str(INPUTS),
            "--out-dir", str(OUT),
            "--only", slug,
            "--candidates", str(args.candidates),
            "--retries", "1",
            "--seed-base", str(SEED_BASE.get(slug, 5000)),
            "--budget", "12",
            "--nar-query-chunk", "1024",
            "--gpu-mem-fraction", "0.90",
            "--max-plan-seconds", "240",
            "--intro-bars", "4" if mode in ("atmospheric", "epic", "epic_rap") else "2",
            "--melody-target", "zh-pop",
            "--hook-max-tile", "3.5",
            "--no-early-accept",
            *shape_args(mode),
        ]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOGS / f"zh_pop_{slug}_{stamp}.log"
        status["titles"][slug] = {
            "state": "running", "mode": mode, "started": stamp,
            "log": str(log_path), "command": cmd,
        }
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {slug} ({mode}) -> {log_path.name}",
              flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace")
        ok = proc.returncode == 0 and selected.exists()
        status["titles"][slug].update({
            "state": "done" if ok else "failed",
            "returncode": proc.returncode,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {slug}: "
              f"{'done' if ok else 'FAILED rc=' + str(proc.returncode)}", flush=True)

    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
