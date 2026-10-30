from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YUE_ROOT = PROJECT_ROOT / "vendor" / "YuE"
YUE_INFER = YUE_ROOT / "inference" / "infer.py"
YUE_PYTHON = PROJECT_ROOT / "vendor" / "ACE-Step-1.5" / ".venv" / "Scripts" / "python.exe"
MODEL_ROOT = PROJECT_ROOT / "models" / "yue"
STAGE1 = MODEL_ROOT / "YuE-s1-7B-anneal-en-cot"
STAGE2 = MODEL_ROOT / "YuE-s2-1B-general"
XCODEC = YUE_ROOT / "inference" / "xcodec_mini_infer"


def build_command(genre: Path, lyrics: Path, output: Path, segments: int = 1,
                  max_new_tokens: int = 1200, seed: int = 42,
                  audio_prompt: Path | None = None) -> list[str]:
    genre = genre.resolve()
    lyrics = lyrics.resolve()
    output = output.resolve()
    if audio_prompt:
        audio_prompt = audio_prompt.resolve()
    required = [YUE_INFER, STAGE1 / "config.json", STAGE1 / "model.safetensors.index.json",
                STAGE2 / "config.json", STAGE2 / "model.safetensors", XCODEC / "final_ckpt" / "ckpt_00360000.pth"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("YuE is not ready; missing: " + ", ".join(missing))
    command = [str(YUE_PYTHON), str(YUE_INFER), "--stage1_model", str(STAGE1),
               "--stage2_model", str(STAGE2), "--genre_txt", str(genre),
               "--lyrics_txt", str(lyrics), "--output_dir", str(output),
               "--run_n_segments", str(segments), "--stage2_batch_size", "1",
               "--max_new_tokens", str(max_new_tokens), "--repetition_penalty", "1.1",
               "--seed", str(seed), "--basic_model_config", str(XCODEC / "final_ckpt" / "config.yaml"),
               "--resume_path", str(XCODEC / "final_ckpt" / "ckpt_00360000.pth"),
               "--config_path", str(XCODEC / "decoders" / "config.yaml"),
               "--vocal_decoder_path", str(XCODEC / "decoders" / "decoder_131000.pth"),
               "--inst_decoder_path", str(XCODEC / "decoders" / "decoder_151000.pth"),
               "--cuda_idx", "0", "--low_vram"]
    if audio_prompt:
        command.extend(["--use_audio_prompt", "--audio_prompt_path", str(audio_prompt),
                        "--prompt_start_time", "0", "--prompt_end_time", "30"])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated YuE next-generation song backend.")
    parser.add_argument("--genre", type=Path, required=True)
    parser.add_argument("--lyrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "yue_candidates")
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio-prompt", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args.genre, args.lyrics, args.output_dir, args.segments,
                            args.max_new_tokens, args.seed, args.audio_prompt)
    print("Running YuE:", " ".join(command))
    subprocess.run(command, cwd=YUE_ROOT / "inference", check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
