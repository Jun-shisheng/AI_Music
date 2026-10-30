"""Second calibration round: fix the two named references, lengthen the BGM.

Round 1 (run_epic_20261028.py) showed:
  * 03 (御姐 / 哲思史诗) is clean: build-up 16.4 dB, climax at 0.91, coverage 1.08.
  * 01 (空灵) is washed out: coverage 0.51, early peak, 3:58.
  * 02 (战斗) is buried: coverage 0.60, register lift 0.4.
  * 04 (BGM) is only 78 s, but the TSFH NAR LoRA gives it width 0.515.

This round isolates the fixes on the same sampled plans:
  05  same 01 plan, vocal changed from airy/distant to clear/present;
      verses quiet, full orchestra only in the final chorus.
  06  same 02 plan, solo lead vocal, gang/choir pulled back, "no gang vocals".
  07  BGM with the instrumental-cot AR LoRA + TSFH NAR, no intro trim,
      so the plan can breathe to a full-length cue.

One process per title (Windows allocator fragmentation).

Usage:
  vendor/YuE2/.venv/Scripts/python.exe -I tools/generation/run_epic_ab_20261028.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AB_IN = ROOT / "inputs" / "2026-10-28_epic_ab"
BGM_LONG_IN = ROOT / "inputs" / "2026-10-28_bgm_long"
OUT = ROOT / "outputs" / "2026-10-28_epic_ab"
LOGS = ROOT / "logs"
PY = ROOT / "vendor" / "YuE2" / ".venv" / "Scripts" / "python.exe"
BATCH = ROOT / "tools" / "generation" / "yue2_batch_battle.py"
VOCAL_PLANS = ROOT / "outputs" / "2026-10-28_epic" / "candidates"
TSFH = ROOT / "models" / "yue2" / "loras" / "tsfh-steps-from-hell"
INST = ROOT / "models" / "yue2" / "loras" / "instrumental-cot"


def plan_file(prefix: str) -> Path:
    hits = sorted(VOCAL_PLANS.glob(f"{prefix}*/{prefix}_c1_plan_original.abc"))
    if not hits:
        raise SystemExit(f"no plan for {prefix}")
    return hits[0]


TITLES = [
    {"slug": "05_snowline_clear", "inputs": AB_IN, "seed": 7100,
     "plan": plan_file("01_above_the_snowline"), "intro_bars": 4,
     "shapes": ["hook chorus", "thin verse:2"], "final_lift": 2},
    {"slug": "06_battle_solo", "inputs": AB_IN, "seed": 7200,
     "plan": plan_file("02_trade_the_night_for_dawn"), "intro_bars": 2,
     "shapes": ["hook chorus", "thin verse:2"], "final_lift": 2},
    {"slug": "07_break_the_formation_long", "inputs": BGM_LONG_IN, "seed": 7500,
     "plan": None, "intro_bars": 8, "shapes": [], "final_lift": 0,
     "instrumental": True},
]


def build_cmd(title: dict) -> list[str]:
    out_dir = OUT / "bgm" if title.get("instrumental") else OUT
    cmd = [str(PY), "-I", str(BATCH),
           "--inputs", str(title["inputs"]), "--out-dir", str(out_dir),
           "--only", title["slug"], "--candidates", "1", "--retries", "1",
           "--seed-base", str(title["seed"]), "--budget", "12",
           "--nar-query-chunk", "1024", "--gpu-mem-fraction", "0.90",
           "--max-plan-seconds", "240", "--intro-bars", str(title["intro_bars"]),
           "--no-early-accept"]
    if title.get("plan"):
        cmd += ["--plan-file", str(title["plan"])]
    if title.get("instrumental"):
        cmd += ["--no-diction", "--no-melody",
                "--lora-ar", str(INST / "ar_lora_inst_v3abc.bf16.safetensors"),
                "--lora-nar", str(TSFH / "adapter-nar-194" / "lora.safetensors"),
                "--ar-scale", "1.0", "--nar-scale", "0.375"]
    else:
        cmd += ["--melody-target", "diagnostic", "--hook-max-tile", "3.5"]
        for shape in title["shapes"]:
            cmd += ["--shape", shape]
        if title["final_lift"]:
            cmd += ["--final-lift", str(title["final_lift"])]
    return cmd


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = LOGS / "epic_ab_20261028_status.json"
    status: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "titles": {}}

    for title in TITLES:
        slug = title["slug"]
        out_dir = OUT / "bgm" if title.get("instrumental") else OUT
        selected = out_dir / "selected" / f"{slug}.wav"
        if selected.exists():
            status["titles"][slug] = {"state": "skipped"}
            continue
        cmd = build_cmd(title)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOGS / f"epicab_{slug}_{stamp}.log"
        status["titles"][slug] = {"state": "running", "log": str(log_path),
                                  "command": cmd}
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {slug}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace")
        ok = proc.returncode == 0 and selected.exists()
        status["titles"][slug].update({
            "state": "done" if ok else "failed",
            "returncode": proc.returncode,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
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
