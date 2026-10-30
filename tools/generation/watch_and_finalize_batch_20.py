"""Wait for source generation to finish, then finalize the 20-song batch."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BATCH = PROJECT_ROOT / "outputs" / "batch_20260820"
STATUS = BATCH / "BATCH_STATUS.txt"
FINALIZE_STATUS = BATCH / "FINALIZE_STATUS.txt"
PYTHON = PROJECT_ROOT / "vendor" / "ACE-Step-1.5" / ".venv" / "Scripts" / "python.exe"
FINALIZER = PROJECT_ROOT / "tools" / "generation" / "finalize_batch_20.py"


def main() -> int:
    FINALIZE_STATUS.write_text("waiting for 20 source candidates\n", encoding="utf-8")
    deadline = time.time() + 8 * 60 * 60
    while time.time() < deadline:
        state = STATUS.read_text(encoding="utf-8").strip() if STATUS.exists() else ""
        count = len(list((BATCH / "candidates").glob("*.wav")))
        if state.startswith("finished 20 songs"):
            if count != 20:
                FINALIZE_STATUS.write_text(
                    f"generation finished but only {count}/20 source candidates exist\n",
                    encoding="utf-8",
                )
                return 1
            FINALIZE_STATUS.write_text("source generation complete; finalization running\n", encoding="utf-8")
            result = subprocess.run([str(PYTHON), str(FINALIZER)], cwd=PROJECT_ROOT)
            FINALIZE_STATUS.write_text(
                "finalization complete\n" if result.returncode == 0 else f"finalization failed: exit {result.returncode}\n",
                encoding="utf-8",
            )
            return result.returncode
        time.sleep(30)
    FINALIZE_STATUS.write_text("timed out waiting for source generation\n", encoding="utf-8")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
