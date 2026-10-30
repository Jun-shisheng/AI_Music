from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEART_ROOT = PROJECT_ROOT / "vendor" / "heartlib"
HEART_EXAMPLE = HEART_ROOT / "examples" / "run_music_generation.py"
PYTHON = PROJECT_ROOT / "vendor" / "ACE-Step-1.5" / ".venv" / "Scripts" / "python.exe"
MODEL_ROOT = PROJECT_ROOT / "models" / "heartmula"


def build_command(lyrics: Path, tags: Path, output: Path, seconds: int = 30) -> list[str]:
    lyrics = lyrics.resolve()
    tags = tags.resolve()
    output = output.resolve()
    required = [HEART_EXAMPLE, MODEL_ROOT / "gen_config.json", MODEL_ROOT / "tokenizer.json",
                MODEL_ROOT / "HeartMuLa-oss-3B" / "config.json",
                MODEL_ROOT / "HeartMuLa-oss-3B" / "model.safetensors.index.json",
                MODEL_ROOT / "HeartCodec-oss" / "config.json",
                MODEL_ROOT / "HeartCodec-oss" / "model.safetensors.index.json"]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError("HeartMuLa is not ready; missing: " + ", ".join(missing))
    return [str(PYTHON), str(HEART_EXAMPLE), "--model_path", str(MODEL_ROOT),
            "--version", "3B", "--lyrics", str(lyrics), "--tags", str(tags),
            "--save_path", str(output), "--max_audio_length_ms", str(seconds * 1000),
            "--mula_device", "cuda", "--codec_device", "cuda",
            "--mula_dtype", "bf16", "--codec_dtype", "fp32", "--lazy_load", "true"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a human-listening-first HeartMuLa candidate.")
    parser.add_argument("--lyrics", type=Path, required=True)
    parser.add_argument("--tags", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(args.lyrics, args.tags, args.output, args.seconds)
    print("Running HeartMuLa:", " ".join(command))
    subprocess.run(command, cwd=HEART_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
