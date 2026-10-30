"""Generate a same-seed A/B set across cfg_scale values for listening review.

Uses the 12GB-friendly config discovered on the desktop: codec on GPU, and a
right-sized KV-cache preallocation (the backbone preallocates for max_seq_len,
so 8192 wasted ~2.4 GB at guidance batch 2 and forced cfg=1.0 on the laptop).

Each take is loudness-normalized with a single linear gain so the only
difference the listener hears is the guidance setting, not the level.

Usage:
  python tools/generation/ab_cfg.py --lyrics L.txt --tags T.txt \
      --title starlight --seconds 30 --cfgs 1.0 1.5 2.0 3.0
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

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from heartlib import HeartMuLaGenPipeline  # noqa: E402

import loudness  # noqa: E402


def normalize(src: Path, dst: Path) -> dict:
    """Loudness-match a take (single implementation: tools/generation/loudness.py)."""
    return loudness.normalize_loudness(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lyrics", type=Path, required=True)
    ap.add_argument("--tags", type=Path, required=True)
    ap.add_argument("--title", default="ab")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--cfgs", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--seq-margin", type=int, default=256,
                    help="Extra KV positions above prompt+frames.")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "outputs" / "ab_cfg")
    args = ap.parse_args()

    lyrics = args.lyrics.read_text(encoding="utf-8")
    tags = args.tags.read_text(encoding="utf-8")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pipe = HeartMuLaGenPipeline.from_pretrained(
        str(PROJECT_ROOT / "models" / "heartmula"),
        device={"mula": torch.device("cuda"), "codec": torch.device("cuda")},
        dtype={"mula": torch.bfloat16, "codec": torch.float32},
        version="3B",
        lazy_load=True,
    )

    manifest = []
    for cfg in args.cfgs:
        raw = args.out_dir / f".{args.title}_cfg{cfg}.raw.wav"
        final = args.out_dir / f"{args.title}_cfg{cfg}.wav"
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        t0 = time.time()
        pre = pipe.preprocess({"lyrics": lyrics, "tags": tags}, cfg_scale=cfg)
        prompt_len = int(pre["pos"].shape[-1])
        frames_needed = prompt_len + args.seconds * 1000 // 80 + args.seq_margin
        seq_len = max(1024, -(-frames_needed // 512) * 512)  # round up to 512
        pipe.mula.backbone.max_seq_len = seq_len
        with torch.no_grad():
            out = pipe._forward(pre, max_audio_length_ms=args.seconds * 1000,
                                temperature=args.temperature, topk=args.topk, cfg_scale=cfg)
        frames = out["frames"]
        import torchaudio
        wav = pipe.codec.detokenize(frames.to(pipe.codec.device))
        torchaudio.save(str(raw), wav.to(torch.float32).cpu(), 48000)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        pipe._unload()  # free the codec so mula can reload in RAM-tight VRAM
        normalize(raw, final)
        raw.unlink(missing_ok=True)
        mins = (time.time() - t0) / 60
        dur = sf.info(final).duration
        manifest.append({"cfg": cfg, "file": final.name, "duration_s": round(dur, 2),
                         "frames": int(frames.shape[-1]), "prompt_len": prompt_len,
                         "seq_len": seq_len, "minutes": round(mins, 2),
                         "peak_gb": round(peak_gb, 2)})
        print(f"cfg={cfg}  {dur:.1f}s  frames={frames.shape[-1]} seq={seq_len} "
              f"peak={peak_gb:.2f}GB  {mins:.2f}min -> {final.name}", flush=True)

    (args.out_dir / f"{args.title}_AB_README.txt").write_text(
        "\n".join([
            f"A/B: cfg_scale comparison for {args.title}",
            f"seed={args.seed} temperature={args.temperature} topk={args.topk} "
            f"seconds={args.seconds}",
            "",
            "cfg_scale=1.0 disables classifier-free guidance (batch 1) -- this was",
            "FORCED on the 8GB laptop because batch 2 overflowed VRAM. All previously",
            "delivered songs were generated this way. cfg>1.0 is the library default",
            "(1.5) and is now affordable here after right-sizing the KV preallocation.",
            "All takes are loudness-normalized with a linear gain; only guidance differs.",
            "",
            "Listen at matched loudness and pick the one whose vocal sounds most human",
            "and has the fewest breaks/crackles.",
            "",
        ] + [f"  {m['cfg']:<5} {m['file']}  ({m['duration_s']}s, peak {m['peak_gb']}GB)"
             for m in manifest]), encoding="utf-8")
    (args.out_dir / f"{args.title}_AB_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote", args.out_dir / f"{args.title}_AB_README.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
