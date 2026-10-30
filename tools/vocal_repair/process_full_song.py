"""Full post-processing chain for a generated song: demucs -> humanize chain
(noise floor reduction -> de_buzz EQ -> high-note taming -> frame repair last)
-> remix -> gate.

Usage:
    python process_full_song.py <song.wav> --output <final.wav>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "vocal_repair"))

from humanize_vocal import humanize_stem  # noqa: E402
from stem_pipeline import remix, separate_with_demucs  # noqa: E402
from vocal_quality_gate import analyze_vocals  # noqa: E402


def process(song: Path, output: Path, work_dir: Path | None = None,
            keep_work: bool = False) -> dict:
    work = work_dir or (output.parent / f".work_{song.stem}")
    work.mkdir(parents=True, exist_ok=True)
    vocals, instrumental = separate_with_demucs(song, work)

    humanized = work / "vocals_final.wav"
    humanize_stem(vocals, humanized, variant="de_buzz")

    remix(instrumental, humanized, output)

    gate = analyze_vocals(humanized, humanized)
    result = {
        "output": str(output),
        "gate_decision": gate.decision,
        "gate_score": gate.score,
        "dropout_s": gate.dropout_seconds,
        "harsh_ratio": gate.harsh_frame_ratio,
    }
    print(f"gate: {gate.decision} score={gate.score} dropout={gate.dropout_seconds:.2f}s")

    if not keep_work and work.exists():
        shutil.rmtree(work)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("song", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    result = process(args.song, args.output, args.work_dir, args.keep_work)
    print(f"final -> {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
