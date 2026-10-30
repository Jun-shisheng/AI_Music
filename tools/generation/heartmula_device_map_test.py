"""Test HeartMuLa loading with device_map=auto (GPU+CPU offload) vs cuda.

Purpose: the current workflow uses device_map="cuda" which crashes with
0xC0000005 on this 8 GB GPU (meta-device shard load + CUDA memory pressure).
device_map="auto" worked for YuE Stage 1 on the same GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "heartlib" / "src"))


def main() -> int:
    import torch

    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    print(f"mode: device_map={mode}", flush=True)
    from heartlib.heartmula.configuration_heartmula import HeartMuLaConfig
    from heartlib.heartmula.modeling_heartmula import HeartMuLa

    cfg = HeartMuLaConfig.from_pretrained(r"G:\AI_Music\models\heartmula\HeartMuLa-oss-3B")
    model = HeartMuLa.from_pretrained(
        r"G:\AI_Music\models\heartmula\HeartMuLa-oss-3B",
        config=cfg,
        device_map=mode,
        dtype=torch.bfloat16,
    )
    print("LOADED OK", flush=True)
    print("memory allocated:", torch.cuda.memory_allocated() / 1024**3, "GB", flush=True)
    del model
    torch.cuda.empty_cache()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
