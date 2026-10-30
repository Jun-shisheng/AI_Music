"""Single-config HeartMuLa benchmark: timing + peak VRAM for one sampling setting.

Runs exactly one (cfg, seconds, device) configuration per process so a failing
config cannot poison the others, and so peak-VRAM numbers are clean. Emits a
one-line JSON result on stdout's last line for easy aggregation.

Usage:
  python tools/generation/bench_one.py --lyrics a.txt --tags b.txt \
      --cfg 2.0 --seconds 30 --codec-device cpu --out out.wav
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEART_SRC = PROJECT_ROOT / "vendor" / "heartlib" / "src"
sys.path.insert(0, str(HEART_SRC))


def str2bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "y", "t")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "heartmula")
    ap.add_argument("--lyrics", type=Path, required=True)
    ap.add_argument("--tags", type=Path, required=True)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--codec-device", default="cpu")
    ap.add_argument("--mula-dtype", default="bfloat16")
    ap.add_argument("--codec-dtype", default="float32")
    ap.add_argument("--lazy", type=str2bool, default=True)
    ap.add_argument("--max-seq-len", type=int, default=0,
                    help="Override the backbone KV-cache preallocation (default 8192). "
                         "The cache is sized once and is independent of song length, so a "
                         "smaller value frees VRAM at no quality cost as long as prompt+frames fit.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    from heartlib import HeartMuLaGenPipeline

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    lyrics = args.lyrics.read_text(encoding="utf-8")
    tags = args.tags.read_text(encoding="utf-8")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result: dict = {"cfg": args.cfg, "seconds": args.seconds,
                    "codec_device": args.codec_device, "lazy": args.lazy, "ok": False}

    try:
        t_load = time.time()
        pipe = HeartMuLaGenPipeline.from_pretrained(
            str(args.model_path),
            device={"mula": torch.device("cuda"), "codec": torch.device(args.codec_device)},
            dtype={"mula": dtype_map[args.mula_dtype], "codec": dtype_map[args.codec_dtype]},
            version="3B",
            lazy_load=args.lazy,
        )
        result["load_s"] = round(time.time() - t_load, 1)

        if args.max_seq_len:
            pipe.mula.backbone.max_seq_len = args.max_seq_len
            if hasattr(pipe.mula.backbone, "rope"):
                try:
                    pipe.mula.backbone.rope.max_seq_len = args.max_seq_len
                except Exception:  # noqa: BLE001
                    pass
            result["max_seq_len"] = args.max_seq_len

        t0 = time.time()
        pre = pipe.preprocess({"lyrics": lyrics, "tags": tags}, cfg_scale=args.cfg)
        result["prompt_len"] = int(pre["pos"].shape[-1])
        t1 = time.time()
        with torch.no_grad():
            out = pipe._forward(pre, max_audio_length_ms=args.seconds * 1000,
                                temperature=args.temperature, topk=args.topk,
                                cfg_scale=args.cfg)
        t2 = time.time()
        frames = out["frames"]
        result["frames"] = list(frames.shape)
        result["token_min"] = round((t2 - t1) / 60, 2)
        result["tok_per_s"] = round(frames.shape[-1] / max(t2 - t1, 1e-9), 2)
        result["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        result["reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)

        t3 = time.time()
        import torchaudio
        wav = pipe.codec.detokenize(frames.to(pipe.codec.device))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(args.out), wav.to(torch.float32).cpu(), 48000)
        result["decode_min"] = round((time.time() - t3) / 60, 2)
        result["total_min"] = round((time.time() - t0) / 60, 2)
        result["peak_gb_end"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["result"] = "BENCH"
    print(json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
