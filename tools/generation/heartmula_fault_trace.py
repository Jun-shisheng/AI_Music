"""Run HeartMuLa with faulthandler to capture the crash stack (0xC0000005)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "generation"))

from heartmula_workflow import build_command  # noqa: E402


def main() -> int:
    lyrics = Path("outputs/heartmula_smoke_input/moonlit_lyrics.txt").resolve()
    tags = Path("outputs/heartmula_smoke_input/moonlit_tags.txt").resolve()
    output = Path("outputs/heartmula_candidates/moonlit_hm30s_fault.wav").resolve()
    cmd = build_command(lyrics, tags, output, seconds=30)
    # enable faulthandler in the child process
    cmd = [cmd[0], "-X", "faulthandler", *cmd[1:]]
    env = {**os.environ, "PYTHONFAULTHANDLER": "1"}
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT / "vendor" / "heartlib",
                          env=env, capture_output=True, text=True)
    print("returncode:", proc.returncode)
    print("--- stdout tail ---")
    print(proc.stdout[-3000:])
    print("--- stderr tail ---")
    print(proc.stderr[-3000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
