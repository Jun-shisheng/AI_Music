"""YuE2 generation: style + lyrics -> symbolic plan -> 48 kHz song.

YuE2 is the structurally strongest candidate for the battle/BGM target: it
plans the piece as symbolic ABC notation (melody + chords) before rendering at
25 fps, which is exactly what HeartMuLa's 12.5 Hz codec cannot do. It is also
the riskiest to run here -- upstream asks for a 24 GB GPU, and this box has 12.

The pipeline is told about that explicitly: `memory_budget_gib` caps the
process's CUDA allocation (it also defaults `vae_core_frames` to 512 at <= 12
GiB). `--offload-ar` and `--quantization fp8` are the further escape hatches.

Notes:
- Generation length follows the symbolic plan / lyrics, not a duration flag.
- LoRA adapters (the Two Steps From Hell and instrumental packages) are NOT
  supported by the v0.1.3 inference package, so they need a separate PEFT
  integration step; this runner does not apply them yet.

Usage:
  vendor/YuE2/.venv/Scripts/python.exe tools/generation/yue2_generate.py \
      --style-file inputs/battle_batch/01_dawn_assault_tags.txt \
      --lyrics-file inputs/battle_batch/01_dawn_assault_lyrics.txt \
      --title yue2_dawn --out-dir outputs/yue2_test --budget 12 --offload-ar
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "yue2" / "YuE2-3B"
VAE_DIR = PROJECT_ROOT / "models" / "yue2" / "YuE2-Vae"

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default=None, help="inline style prompt")
    ap.add_argument("--style-file", type=Path, default=None)
    ap.add_argument("--lyrics", default=None)
    ap.add_argument("--lyrics-file", type=Path, default=None)
    ap.add_argument("--title", default="yue2")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--cot", default="full", choices=("full", "melody", "off"))
    ap.add_argument("--cfg-scale", type=float, default=None)
    ap.add_argument("--budget", type=float, default=12.0,
                    help="CUDA memory budget in GiB (upstream default is 24)")
    ap.add_argument("--quantization", default="none", choices=("none", "fp8"))
    ap.add_argument("--offload-ar", action="store_true")
    ap.add_argument("--nar-query-chunk", type=int, default=None,
                    help="chunk size for the NAR prefill attention; required to "
                         "fit a 12 GB card (the high-level API never passes it)")
    ap.add_argument("--backend", default="torch",
                    choices=("torch", "torch-eager", "vllm"))
    ap.add_argument("--model", type=Path, default=MODEL_DIR)
    ap.add_argument("--vae", type=Path, default=VAE_DIR)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "yue2")
    ap.add_argument("--no-hash-check", action="store_true")
    args = ap.parse_args()

    style = args.style or (args.style_file.read_text(encoding="utf-8")
                           if args.style_file else None)
    lyrics = args.lyrics or (args.lyrics_file.read_text(encoding="utf-8")
                             if args.lyrics_file else None)
    if not style or not lyrics:
        ap.error("need --style/--style-file and --lyrics/--lyrics-file")

    # Run this with vendor/YuE2/.venv/Scripts/python.exe -- the package pins
    # torch 2.10.0 and is not importable from the other venvs here.
    from yue2 import YuE2Pipeline

    if args.nar_query_chunk:
        # The NAR stage's SDPA prefill runs one query block the size of the whole
        # piece, which OOMs on a 12 GB card (the protocol forces context=24576
        # and fp8 needs sm89, so chunking is the only lever left). nar.synthesize
        # already supports query_chunk_size, but pipeline.synthesize never passes
        # it -- so inject it here rather than patching the installed package.
        import yue2.nar as nar
        _original = nar.synthesize

        def _chunked(*a, **kw):
            kw.setdefault("query_chunk_size", args.nar_query_chunk)
            return _original(*a, **kw)

        nar.synthesize = _chunked
        print(f"[{time.strftime('%H:%M:%S')}] NAR query chunking enabled: "
              f"{args.nar_query_chunk}", flush=True)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading YuE2 (budget {args.budget} GiB, "
          f"quant={args.quantization}, offload_ar={args.offload_ar}) ...", flush=True)
    pipe = YuE2Pipeline(
        str(args.model), str(args.vae), device="cuda",
        memory_budget_gib=args.budget, quantization=args.quantization,
        offload_ar=args.offload_ar, backend=args.backend,
        verify_hashes=not args.no_hash_check,
    )
    print(f"[{time.strftime('%H:%M:%S')}] loaded in {time.time() - t0:.0f}s", flush=True)

    kwargs = {"cot": args.cot, "seed": args.seed}
    if args.cfg_scale is not None:
        kwargs["cfg_scale"] = args.cfg_scale
    tg = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] generating ...", flush=True)
    song = pipe(style=style, lyrics=lyrics, **kwargs)

    wav = out_dir / f"{args.title}.wav"
    song.save(str(wav))
    song.save_artifacts(str(out_dir / f"{args.title}_artifacts"))
    meta = {
        "backend": "yue2-3b", "title": args.title, "seed": args.seed,
        "cot": args.cot, "budget_gib": args.budget,
        "quantization": args.quantization, "offload_ar": args.offload_ar,
        "load_seconds": round(t0 and (time.time() - tg), 1),
        "generate_seconds": round(time.time() - tg, 1),
        "audio": wav.name,
        "timing": getattr(song, "timing", None),
    }
    (out_dir / f"{args.title}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] done: {wav} "
          f"({time.time() - tg:.0f}s generation)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
