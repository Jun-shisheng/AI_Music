"""Pre-compute CPU Demucs stems for a list of songs into the Voice Studio cache.

The SVC cover job looks up its separation cache by a hash of the source audio,
so this only warms that exact cache; the cover command later reuses it instead
of spending another CPU pass. Resumable: completed entries are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.voice_studio import workflow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.list.read_text(encoding="utf-8"))
    args.log_dir.mkdir(parents=True, exist_ok=True)
    context = SimpleNamespace(
        workspace=ROOT / "data/voice_studio",
        audio_python=ROOT / "tools/vocal_repair/.venv/Scripts/python.exe",
        separation_device="cpu",
    )
    for entry in entries:
        source = (ROOT / entry["source"]).resolve()
        if not source.is_file():
            print(f"SKIP {entry['slug']}: missing source {source}", flush=True)
            continue
        vocals, instrumental = workflow.separate(
            context, source, args.log_dir / entry["slug"])
        print(f"OK {entry['slug']}: {vocals} | {instrumental}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
