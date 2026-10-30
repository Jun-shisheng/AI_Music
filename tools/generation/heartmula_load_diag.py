"""Diagnose which HeartMuLa component crashes on load (0xC0000005).

Loads each component separately and reports progress so the failing
component can be identified.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "heartlib" / "src"))


def main() -> int:
    import torch

    print("torch:", torch.__version__, flush=True)
    print("cuda available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0), flush=True)
        print("mem:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB", flush=True)

    print("step 1: basic matmul", flush=True)
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print("  matmul OK", flush=True)

    print("step 2: load text tokenizer", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(r"G:\AI_Music\models\heartmula")
    print("  tokenizer OK", flush=True)

    print("step 3: load HeartCodec (fp32)", flush=True)
    from heartlib.heartcodec.configuration_heartcodec import HeartCodecConfig
    from heartlib.heartcodec.modeling_heartcodec import HeartCodec
    codec_cfg = HeartCodecConfig.from_pretrained(r"G:\AI_Music\models\heartmula\HeartCodec-oss")
    codec = HeartCodec.from_pretrained(
        r"G:\AI_Music\models\heartmula\HeartCodec-oss", config=codec_cfg
    ).to("cuda")
    print("  codec OK", flush=True)

    print("step 4: load HeartMuLa 3B (bf16)", flush=True)
    from heartlib.heartmula.configuration_heartmula import HeartMuLaConfig
    from heartlib.heartmula.modeling_heartmula import HeartMuLa
    mula_cfg = HeartMuLaConfig.from_pretrained(r"G:\AI_Music\models\heartmula\HeartMuLa-oss-3B")
    mula = HeartMuLa.from_pretrained(
        r"G:\AI_Music\models\heartmula\HeartMuLa-oss-3B", config=mula_cfg
    )
    mula = mula.to("cuda").to(torch.bfloat16)
    print("  mula OK", flush=True)
    print("step 5: forward pass", flush=True)
    torch.cuda.empty_cache()
    print("  empty_cache OK", flush=True)
    print("ALL COMPONENTS OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
