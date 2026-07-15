"""Download ACE-Step 1.5 model weights (~10GB).

Uses ACE-Step's built-in downloader. Auto-detects HuggingFace vs ModelScope
based on network (ModelScope preferred for China).

Run standalone or via start.bat.
"""

import subprocess
import sys
from pathlib import Path

ACESTEP_DIR = Path("G:/AI_Music/vendor/ACE-Step-1.5")
CHECKPOINTS_DIR = ACESTEP_DIR / "checkpoints"

MAIN_COMPONENTS = [
    "acestep-v15-turbo",
    "vae",
    "Qwen3-Embedding-0.6B",
    "acestep-5Hz-lm-1.7B",
]


def _has_weights(model_path: Path) -> bool:
    """Check if model directory contains weight files."""
    weight_patterns = [
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "diffusion_pytorch_model.safetensors",
    ]
    if not model_path.is_dir():
        return False
    return any((model_path / p).exists() for p in weight_patterns)


def models_exist() -> bool:
    """Check if all main model components exist with weights."""
    for component in MAIN_COMPONENTS:
        if not _has_weights(CHECKPOINTS_DIR / component):
            return False
    return True


def download() -> bool:
    """Run acestep-download to fetch model weights. Returns True on success."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACE-Step Model Download")
    print(f"Checkpoints: {CHECKPOINTS_DIR}")
    print("Source: auto-detect (HuggingFace / ModelScope)")
    print("Size: ~10GB")
    print("=" * 60)
    print()

    cmd = [
        str(ACESTEP_DIR / ".venv" / "Scripts" / "python.exe"),
        "-m", "acestep.model_downloader",
        "--dir", str(CHECKPOINTS_DIR),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ACESTEP_DIR),
            env={**__import__("os").environ, "ACESTEP_CHECKPOINTS_DIR": str(CHECKPOINTS_DIR)},
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


if __name__ == "__main__":
    if models_exist():
        print("[OK] All models already downloaded.")
        sys.exit(0)

    print("[INFO] Models not found. Starting download...")
    success = download()
    if success:
        print("[OK] Download complete.")
        sys.exit(0)
    else:
        print("[ERROR] Download failed. Check network or try again.")
        sys.exit(1)
