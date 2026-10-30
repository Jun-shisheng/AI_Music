"""One-shot ACE-Step 1.5 generation, source-clean (no post-processing).

ACE-Step 1.5 is the second local backend next to HeartMuLa. It differs in two
ways that matter for the battle/BGM target:

  * conditioning is a free-form `caption` plus optional musical metadata
    (bpm/key/time signature) instead of HeartMuLa's comma-separated tag soup;
  * it decodes at 25 Hz latents rather than HeartCodec's 12.5 Hz frames, which
    is the structural reason HeartMuLa smears fast transients.

Like clean_pipeline.py, nothing here touches the audio after the model writes
it: loudness normalization is a separate delivery step.

Run with the ACE venv (it has the acestep package and its dependencies):
    vendor/ACE-Step-1.5/.venv/Scripts/python.exe tools/generation/acestep_generate.py \
        --caption "..." --duration 60 --out-dir outputs/battle_test
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ACE-Step logs carry emoji/box-drawing characters and this process is launched
# with a redirected (GBK) console; without this, printing an init message raises
# UnicodeEncodeError and kills the run after the models have already loaded.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ACE_ROOT = PROJECT_ROOT / "vendor" / "ACE-Step-1.5"
CHECKPOINTS = ACE_ROOT / "checkpoints"

# `acestep` lives in the vendored repo, not in site-packages.
sys.path.insert(0, str(ACE_ROOT))

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import (  # noqa: E402
    GenerationConfig,
    GenerationParams,
    generate_music,
)
from acestep.llm_inference import LLMHandler  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # A job file is the reliable way to pass a long caption on Windows: shell
    # quoting through bash -> PowerShell -> Start-Process mangles spaces and
    # newlines. Everything the CLI can set, the JSON can set too.
    ap.add_argument("--job", type=Path,
                    help="JSON file of arguments; explicit CLI flags win")
    ap.add_argument("--caption", default=None,
                    help="free-form music description (<=512 chars)")
    ap.add_argument("--lyrics", default="[Instrumental]",
                    help="lyrics text, or [Instrumental]")
    ap.add_argument("--instrumental", action="store_true")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="target seconds (10-600)")
    ap.add_argument("--bpm", type=int, default=None)
    ap.add_argument("--vocal-language", dest="vocal_language", default="unknown",
                    help="ISO 639-1 code for the sung language (zh, en, ja ...)")
    ap.add_argument("--key", dest="keyscale", default="")
    ap.add_argument("--time-signature", dest="timesignature", default="",
                    help="2, 3, 4 or 6 (empty = LM decides)")
    ap.add_argument("--title", default="ace")
    ap.add_argument("--seed", type=int, default=-1,
                    help="-1 = random; otherwise reproducible")
    ap.add_argument("--batch", type=int, default=1, help="candidates per run (1-8)")
    ap.add_argument("--steps", type=int, default=8, help="turbo: 1-20, 8 recommended")
    ap.add_argument("--shift", type=float, default=3.0, help="turbo wants 3.0")
    ap.add_argument("--dit", default="acestep-v15-turbo")
    ap.add_argument("--lm", default="acestep-5Hz-lm-0.6B")
    ap.add_argument("--lm-backend", default="pt", help="vllm is not installable on Windows")
    ap.add_argument("--no-lm", action="store_true",
                    help="skip the 5Hz LM (faster, lower quality)")
    ap.add_argument("--offload", action="store_true",
                    help="CPU-offload weights (needed on a 12 GB card)")
    ap.add_argument("--audio-format", default="wav")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "outputs" / "ace")

    # Load the job file first so its values become the parser defaults; any
    # flag given on the command line then overrides it.
    argv = sys.argv[1:]
    job: dict = {}
    if "--job" in argv:
        job_path = Path(argv[argv.index("--job") + 1])
        job = json.loads(job_path.read_text(encoding="utf-8"))
        known = {a.dest for a in ap._actions}
        unknown = sorted(set(job) - known)
        if unknown:
            ap.error(f"unknown key(s) in job file: {', '.join(unknown)}")
    ap.set_defaults(**job)
    args = ap.parse_args()
    args.out_dir = Path(args.out_dir)
    if not args.caption:
        ap.error("--caption is required (via CLI or job file)")
    return args


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading DiT {args.dit} ...", flush=True)
    dit = AceStepHandler()
    msg, ok = dit.initialize_service(
        project_root=str(ACE_ROOT),
        config_path=args.dit,
        device="cuda",
        offload_to_cpu=args.offload,
        offload_dit_to_cpu=args.offload,
    )
    print(f"[{time.strftime('%H:%M:%S')}] dit init ok={ok}: {msg}", flush=True)
    if not ok:
        return 1

    llm = LLMHandler()
    if not args.no_lm:
        print(f"[{time.strftime('%H:%M:%S')}] loading LM {args.lm} ({args.lm_backend}) ...",
              flush=True)
        msg, ok = llm.initialize(
            checkpoint_dir=str(CHECKPOINTS),
            lm_model_path=args.lm,
            backend=args.lm_backend,
            device="cuda",
            offload_to_cpu=args.offload,
        )
        print(f"[{time.strftime('%H:%M:%S')}] lm init ok={ok}: {msg}", flush=True)
        if not ok:
            return 1

    params = GenerationParams(
        caption=args.caption,
        lyrics=args.lyrics,
        instrumental=args.instrumental,
        duration=args.duration,
        bpm=args.bpm,
        keyscale=args.keyscale,
        timesignature=args.timesignature,
        vocal_language=args.vocal_language,
        thinking=not args.no_lm,
        inference_steps=args.steps,
        shift=args.shift,
        seed=args.seed,
    )
    config = GenerationConfig(
        batch_size=args.batch,
        audio_format=args.audio_format,
        use_random_seed=args.seed < 0,
    )

    print(f"[{time.strftime('%H:%M:%S')}] generating {args.batch} x {args.duration:.0f}s ...",
          flush=True)
    result = generate_music(dit, llm, params, config, save_dir=str(out_dir))
    if not result.success:
        print(f"[{time.strftime('%H:%M:%S')}] FAILED: {result.error}", flush=True)
        return 1

    takes = []
    for i, audio in enumerate(result.audios, start=1):
        src = Path(audio["path"])
        dst = out_dir / f"{args.title}_c{i}{src.suffix}"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        seed = (audio.get("params") or {}).get("seed")
        takes.append({"file": dst.name, "seed": seed})
        print(f"[{time.strftime('%H:%M:%S')}] take {i}: {dst.name} seed={seed}", flush=True)

    manifest = {
        "backend": "ace-step-1.5",
        "dit": args.dit,
        "lm": None if args.no_lm else args.lm,
        "caption": args.caption,
        "lyrics": args.lyrics,
        "instrumental": args.instrumental,
        "duration": args.duration,
        "bpm": args.bpm,
        "keyscale": args.keyscale,
        "timesignature": args.timesignature,
        "vocal_language": args.vocal_language,
        "inference_steps": args.steps,
        "shift": args.shift,
        "offload": args.offload,
        "elapsed_s": round(time.time() - t0, 1),
        "takes": takes,
    }
    (out_dir / f"{args.title}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] done in {time.time() - t0:.0f}s -> {out_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
