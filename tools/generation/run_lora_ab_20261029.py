"""Does the TSFH NAR LoRA widen YuE2's stereo image without eating the vocal?

Context: the 10-29 film batch moved arrangement and dynamics but the stereo
width stayed at 0.25-0.37 against 0.50 in the references.  Repeating "wide
stereo" in the style tags did nothing, so the next lever is the renderer:
`tsfh-steps-from-hell/adapter-nar-194` was trained on TSFH orchestra and the
instrumental BGM run with it measured 0.515 wide.

This is a controlled test: same title, same seed, same tags, same plan shapes;
only the NAR LoRA scale changes.  AR is untouched (scale 0) so the sampled
plan is identical across variants.

Variants:
  ctrl    no LoRA                     (baseline reproducibility check)
  nar015  NAR LoRA at 0.15
  nar025  NAR LoRA at 0.25

One render at a time; the CUDA allocator does not survive two YuE2 processes
on a 12 GB card.

Usage:
  vendor/YuE2/.venv/Scripts/python.exe -I tools/generation/run_lora_ab_20261029.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "inputs" / "2026-10-29_film_ref"
OUT = ROOT / "outputs" / "2026-10-29_lora_ab"
LOGS = ROOT / "logs"
PY = ROOT / "vendor" / "YuE2" / ".venv" / "Scripts" / "python.exe"
BATCH = ROOT / "tools" / "generation" / "yue2_batch_battle.py"
QA_PY = ROOT / "tools" / "vocal_repair" / ".venv" / "Scripts" / "python.exe"
STYLE_QA = ROOT / "tools" / "generation" / "style_qa.py"
LORA_NAR = (ROOT / "models" / "yue2" / "loras" / "tsfh-steps-from-hell"
            / "adapter-nar-194" / "lora.safetensors")

SLUG = "03_names_worn_away"
SEED = 9300
VARIANTS = [
    ("ctrl", None, 0.0),
    ("nar015", LORA_NAR, 0.15),
    ("nar025", LORA_NAR, 0.25),
]


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = LOGS / "lora_ab_20261029_status.json"
    status: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "variants": {}}

    for name, lora, scale in VARIANTS:
        out_dir = OUT / name
        selected = out_dir / "selected" / f"{SLUG}.wav"
        if selected.exists():
            print(f"[{time.strftime('%H:%M:%S')}] {name}: already done", flush=True)
            continue
        cmd = [
            str(PY), "-I", str(BATCH),
            "--inputs", str(INPUTS), "--out-dir", str(out_dir),
            "--only", SLUG, "--candidates", "1", "--retries", "1",
            "--seed-base", str(SEED), "--budget", "12",
            "--nar-query-chunk", "1024", "--gpu-mem-fraction", "0.90",
            "--max-plan-seconds", "215", "--intro-bars", "8",
            "--melody-target", "diagnostic", "--hook-max-tile", "3.5",
            "--no-early-accept",
            "--shape", "hook chorus", "--shape", "thin verse:2",
            "--final-lift", "0",
        ]
        if lora is not None:
            cmd += ["--lora-nar", str(lora), "--nar-scale", str(scale)]

        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOGS / f"lora_ab_{name}_{stamp}.log"
        status["variants"][name] = {"state": "running", "log": str(log_path),
                                     "command": cmd}
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {name} (nar scale {scale})", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace")
        entry = {"returncode": proc.returncode,
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        if proc.returncode == 0 and selected.exists():
            entry["state"] = "done"
            qa = subprocess.run(
                [str(QA_PY), "-I", str(STYLE_QA), "--file", str(selected),
                 "--json", str(out_dir / f"{name}_style.json")],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            entry["style_qa"] = qa.stdout.strip().splitlines()[-3:]
            result = next(iter(out_dir.glob(f"candidates/{SLUG}_*/result.json")), None)
            if result:
                try:
                    data = json.loads(result.read_text(encoding="utf-8"))
                    entry.update({
                        "duration_s": data["arc"]["duration_s"],
                        "arc": data["arc"]["verdict"],
                        "lift_db": data["arc"]["lift_db"],
                        "peak_at": data["arc"]["peak_at"],
                        "coverage": data.get("diction"),
                    })
                except Exception:
                    pass
        else:
            entry["state"] = "failed"
        status["variants"][name].update(entry)
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {entry['state']} "
              f"{entry.get('arc','')} cov={entry.get('coverage')}", flush=True)

    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
