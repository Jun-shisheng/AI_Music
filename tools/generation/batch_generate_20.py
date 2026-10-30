"""Generate the prepared 20-song sample batch with isolated HeartMuLa runs."""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / "vendor" / "ACE-Step-1.5" / ".venv" / "Scripts" / "python.exe"
PIPELINE = PROJECT_ROOT / "tools" / "generation" / "clean_pipeline.py"
INPUT_ROOT = PROJECT_ROOT / "outputs" / "batch_20260820" / "inputs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "batch_20260820" / "candidates"
RUN_ROOT = OUTPUT_ROOT / "_runs"
STATUS = PROJECT_ROOT / "outputs" / "batch_20260820" / "BATCH_STATUS.txt"


def read_index() -> list[tuple[str, str]]:
    rows = []
    for line in (INPUT_ROOT / "INDEX.txt").read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            slug, title, _style = line.split("\t", 2)
            rows.append((slug, title))
    return rows


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows = read_index()
    STATUS.write_text(f"started {len(rows)} songs\n", encoding="utf-8")

    for index, (slug, title) in enumerate(rows):
        final = OUTPUT_ROOT / f"{slug}.wav"
        if final.exists():
            continue
        run_dir_before = set(RUN_ROOT.glob(f"{slug}_*"))
        seed = 520 + index * 11
        temperature = [0.82, 0.94, 1.06, 0.88][index % 4]
        topk = [35, 50, 65, 45][index % 4]
        start = time.time()
        STATUS.write_text(
            f"running {index + 1}/{len(rows)} {slug} seed={seed}\n",
            encoding="utf-8",
        )
        cmd = [
            str(PYTHON), str(PIPELINE),
            "--lyrics", str(INPUT_ROOT / f"{slug}_lyrics.txt"),
            "--tags", str(INPUT_ROOT / f"{slug}_tags.txt"),
            "--title", slug,
            "--candidates", "1",
            "--seconds", "30",
            "--base-seed", str(seed),
            "--out-dir", str(RUN_ROOT),
        ]
        env = dict(__import__("os").environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        try:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
            new_runs = sorted(
                set(RUN_ROOT.glob(f"{slug}_*")) - run_dir_before,
                key=lambda p: p.stat().st_mtime,
            )
            if not new_runs:
                raise RuntimeError("clean_pipeline completed without a run directory")
            produced = new_runs[-1] / f"{slug}_c1.wav"
            if not produced.exists():
                raise RuntimeError(f"missing generated WAV: {produced}")
            shutil.copy2(produced, final)
            shutil.rmtree(new_runs[-1])
            elapsed = (time.time() - start) / 60
            STATUS.write_text(
                f"completed {index + 1}/{len(rows)} {slug} {title} {elapsed:.1f} min\n",
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            STATUS.write_text(
                f"failed {index + 1}/{len(rows)} {slug}: {exc}\n",
                encoding="utf-8",
            )
    if RUN_ROOT.exists() and not any(RUN_ROOT.iterdir()):
        RUN_ROOT.rmdir()
    STATUS.write_text(f"finished {len(rows)} songs\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
