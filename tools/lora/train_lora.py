"""ACE-Step LoRA training wrapper for AI_Music curated datasets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACE = Path(os.environ.get("ACESTEP_DIR", PROJECT_ROOT / "vendor" / "ACE-Step-1.5"))
DEFAULT_CHECKPOINTS = Path(
    os.environ.get("ACESTEP_CHECKPOINT_DIR", DEFAULT_ACE / "checkpoints")
)
DEFAULT_DATASET = PROJECT_ROOT / "data" / "lora_datasets" / "from_top_songs"
# Side-Step refuses paths outside the ACE-Step repo root ("Path escapes safe root").
DEFAULT_TENSORS = Path(
    os.environ.get("ACESTEP_LORA_TENSOR_DIR", DEFAULT_ACE / "lora_workspace" / "tensors" / "from_top_songs")
)
DEFAULT_OUTPUT = Path(
    os.environ.get("ACESTEP_LORA_OUTPUT_DIR", DEFAULT_ACE / "lora_workspace" / "output" / "from_top_songs")
)


def _python_for_ace(ace_dir: Path) -> str:
    # Prefer uv/project env inside ACE-Step if present
    for candidate in (
        ace_dir / ".venv" / "Scripts" / "python.exe",
        ace_dir / ".venv" / "python.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def build_preprocess_cmd(
    *,
    ace_dir: Path,
    checkpoint_dir: Path,
    dataset_json: Path,
    audio_dir: Path,
    tensor_output: Path,
    model_variant: str,
    max_duration: float | None = None,
) -> list[str]:
    py = _python_for_ace(ace_dir)
    cmd = [
        py,
        str(ace_dir / "train.py"),
        "--yes",
        "fixed",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--model-variant",
        model_variant,
        "--preprocess",
        "--audio-dir",
        str(audio_dir),
        "--dataset-json",
        str(dataset_json),
        "--tensor-output",
        str(tensor_output),
        "--dataset-dir",
        str(tensor_output),
        "--output-dir",
        str(tensor_output),
    ]
    if max_duration is not None:
        cmd.extend(["--max-duration", str(max_duration)])
    return cmd


def build_train_cmd(
    *,
    ace_dir: Path,
    checkpoint_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    model_variant: str,
    epochs: int,
    rank: int,
    learning_rate: float,
    minimal_vram: bool = False,
    optimizer_type: str = "adamw",
    gradient_accumulation: int = 1,
) -> list[str]:
    py = _python_for_ace(ace_dir)
    cmd = [
        py,
        str(ace_dir / "train.py"),
        "--yes",
        "fixed",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--model-variant",
        model_variant,
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--rank",
        str(rank),
        "--lr",
        str(learning_rate),
        "--batch-size",
        "1",
        "--gradient-checkpointing",
        "--optimizer-type",
        optimizer_type,
        "--gradient-accumulation",
        str(gradient_accumulation),
    ]
    if minimal_vram:
        cmd.append("--offload-encoder")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess / train ACE-Step LoRA from exported dataset.")
    parser.add_argument("--ace-dir", type=Path, default=DEFAULT_ACE)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tensor-dir", type=Path, default=DEFAULT_TENSORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-variant", default="turbo", help="turbo/base/sft or folder name under checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-duration", type=float, default=None, help="Clip audio during preprocess (seconds)")
    parser.add_argument(
        "--minimal-vram",
        action="store_true",
        help="8GB-friendly defaults: rank 16, offload-encoder, max-duration 45, more grad accum",
    )
    parser.add_argument("--optimizer-type", default="adamw", choices=["adamw", "adamw8bit", "adafactor", "prodigy"])
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.minimal_vram:
        if args.rank == 64:
            args.rank = 16
        if args.max_duration is None:
            args.max_duration = 45.0
        if args.gradient_accumulation == 1:
            args.gradient_accumulation = 4
        if args.epochs == 100:
            args.epochs = 40
        print(
            "Minimal VRAM mode: "
            f"rank={args.rank}, max_duration={args.max_duration}, "
            f"grad_accum={args.gradient_accumulation}, epochs={args.epochs}, offload-encoder"
        )

    dataset_json = args.dataset_dir / "dataset.json"
    audio_dir = args.dataset_dir / "audio"
    if not dataset_json.exists():
        print(f"Missing {dataset_json}. Run: python tools/lora/export_dataset.py")
        return 1

    cmds: list[list[str]] = []
    if not args.train_only:
        cmds.append(
            build_preprocess_cmd(
                ace_dir=args.ace_dir,
                checkpoint_dir=args.checkpoint_dir,
                dataset_json=dataset_json,
                audio_dir=audio_dir,
                tensor_output=args.tensor_dir,
                model_variant=args.model_variant,
                max_duration=args.max_duration,
            )
        )
    if not args.preprocess_only:
        cmds.append(
            build_train_cmd(
                ace_dir=args.ace_dir,
                checkpoint_dir=args.checkpoint_dir,
                dataset_dir=args.tensor_dir,
                output_dir=args.output_dir,
                model_variant=args.model_variant,
                epochs=args.epochs,
                rank=args.rank,
                learning_rate=args.lr,
                minimal_vram=args.minimal_vram,
                optimizer_type=args.optimizer_type,
                gradient_accumulation=args.gradient_accumulation,
            )
        )

    for cmd in cmds:
        print(" ".join(cmd))
        if args.dry_run:
            continue
        args.tensor_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(cmd, cwd=str(args.ace_dir))
        if completed.returncode != 0:
            print(
                "Training command failed. Ensure peft/lightning are installed in ACE-Step env:\n"
                "  pip install peft lightning bitsandbytes\n"
                "Official LoRA tutorial recommends >=16GB VRAM.\n"
                "On RTX 4060 8GB try: python tools/lora/train_lora.py --minimal-vram\n"
                "Also keep --dataset-dir/--output-dir under vendor/ACE-Step-1.5 (safe root)."
            )
            return completed.returncode

    adapter_hint = args.output_dir / "final"
    if not adapter_hint.exists():
        adapter_hint = args.output_dir
    has_adapter = any(args.output_dir.glob("**/*")) if args.output_dir.exists() else False
    if args.dry_run:
        print("Dry-run only; no training executed.")
    else:
        print(f"Done. LoRA output dir: {adapter_hint}")
        if not has_adapter and not args.preprocess_only:
            print("WARNING: output dir looks empty — training may have failed (check ACE-Step logs).")
            return 1
        if not args.preprocess_only:
            print(f"Set ACESTEP_LORA_PATH={adapter_hint.as_posix()} and restart API.")
        else:
            print("Preprocess finished. Next: python tools/lora/train_lora.py --train-only --minimal-vram")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
