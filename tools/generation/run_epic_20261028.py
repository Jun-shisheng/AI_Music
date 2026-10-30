"""Calibration run for the 2026-10-28 epic / ethereal line.

The 2026-10-26 Chinese batch was rejected by ear as 糖水歌.  This driver renders
one take per archetype so the user can pick a direction before a full batch:

  01 空灵史诗   epic_ethereal  A minor  74  -- 诀爱型：辽阔、气声、长音、末副歌才满编
  02 战斗史诗   battle_anthem  D minor 152  -- 新·与天齐型：口号、战鼓、民乐、齐唱
  03 哲思史诗   epic           A minor 100  -- 自由的你型：一个比喻、桥段反转、御姐
  04 战斗 BGM   instrumental   D minor 148  -- TSFH LoRA（AR 0.5 / NAR 0.375）

One process per title: the CUDA caching allocator fragments across titles on
Windows and `expandable_segments` is not supported there.

Usage:
  vendor/YuE2/.venv/Scripts/python.exe -I tools/generation/run_epic_20261028.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VOCAL_IN = ROOT / "inputs" / "2026-10-28_epic"
BGM_IN = ROOT / "inputs" / "2026-10-28_bgm"
OUT = ROOT / "outputs" / "2026-10-28_epic"
LOGS = ROOT / "logs"
PY = ROOT / "vendor" / "YuE2" / ".venv" / "Scripts" / "python.exe"
BATCH = ROOT / "tools" / "generation" / "yue2_batch_battle.py"
STYLE_QA = ROOT / "tools" / "generation" / "style_qa.py"
QA_PY = ROOT / "tools" / "vocal_repair" / ".venv" / "Scripts" / "python.exe"
LORA = ROOT / "models" / "yue2" / "loras" / "tsfh-steps-from-hell"

TITLES = [
    {"slug": "01_above_the_snowline", "mode": "epic_ethereal",
     "inputs": VOCAL_IN, "seed": 7100, "intro_bars": 4},
    {"slug": "02_trade_the_night_for_dawn", "mode": "battle_anthem",
     "inputs": VOCAL_IN, "seed": 7200, "intro_bars": 2},
    {"slug": "03_the_rope_end", "mode": "epic",
     "inputs": VOCAL_IN, "seed": 7300, "intro_bars": 4},
    {"slug": "01_break_the_formation", "mode": "instrumental_bgm",
     "inputs": BGM_IN, "seed": 7400, "intro_bars": 4},
]


def shape_args(mode: str) -> list[str]:
    if mode == "instrumental_bgm":
        return []
    args = ["--shape", "hook chorus", "--shape", "thin verse:2",
            "--final-lift", "2"]
    return args


def build_cmd(title: dict) -> list[str]:
    mode = title["mode"]
    out_dir = OUT / "bgm" if mode == "instrumental_bgm" else OUT
    cmd = [
        str(PY), "-I", str(BATCH),
        "--inputs", str(title["inputs"]),
        "--out-dir", str(out_dir),
        "--only", title["slug"],
        "--candidates", "1",
        "--retries", "1",
        "--seed-base", str(title["seed"]),
        "--budget", "12",
        "--nar-query-chunk", "1024",
        "--gpu-mem-fraction", "0.90",
        "--max-plan-seconds", "240",
        "--intro-bars", str(title["intro_bars"]),
        "--no-early-accept",
    ]
    if mode == "instrumental_bgm":
        cmd += [
            "--no-diction", "--no-melody",
            "--lora-ar", str(LORA / "adapter-ar-195" / "lora.safetensors"),
            "--lora-nar", str(LORA / "adapter-nar-194" / "lora.safetensors"),
            "--ar-scale", "0.5", "--nar-scale", "0.375",
        ]
    else:
        cmd += ["--melody-target", "diagnostic", "--hook-max-tile", "3.5",
                *shape_args(mode)]
    return cmd


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = LOGS / "epic_20261028_status.json"
    status: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "titles": {}}
    only = sys.argv[1] if len(sys.argv) > 1 else ""

    for title in TITLES:
        slug = title["slug"]
        if only and only not in slug:
            continue
        out_dir = OUT / "bgm" if title["mode"] == "instrumental_bgm" else OUT
        selected = out_dir / "selected" / f"{slug}.wav"
        if selected.exists():
            status["titles"][slug] = {"state": "skipped", "reason": "already promoted"}
            continue
        cmd = build_cmd(title)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOGS / f"epic_{slug}_{stamp}.log"
        status["titles"][slug] = {"state": "running", "mode": title["mode"],
                                  "started": stamp, "log": str(log_path),
                                  "command": cmd}
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {slug} ({title['mode']}) -> "
              f"{log_path.name}", flush=True)
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
            entry["style_qa_stdout"] = qa.stdout.strip()[-2000:]
        status["titles"][slug].update(entry)
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
