"""Single-variant GPU worker. Invoke through ``reference_studio render``."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

# -I deliberately excludes cwd; only this known project is added.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.reference_studio.workflow import checked_manifest, rendering_config, write_json


def generation_kwargs(experiment: Path, manifest: dict, variant: str) -> dict:
    target = manifest["variants"][variant]
    rendering = rendering_config(manifest)
    kwargs = {"style": (experiment/"style.txt").read_text(encoding="utf-8"),
              "lyrics": (experiment/"lyrics.txt").read_text(encoding="utf-8"),
              "abc": (experiment/target["plan"]).read_text(encoding="utf-8"),
              "cot": rendering["cot"], "seed": target["seed"]}
    if rendering["cfg_scale"] is not None:
        kwargs["cfg_scale"] = rendering["cfg_scale"]
    if rendering["semantic_max_tokens"] is not None:
        kwargs["semantic_sampling"] = {"max_tokens": rendering["semantic_max_tokens"]}
    return kwargs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--variant", required=True, choices=("baseline", "sparse", "full", "arranged"))
    ap.add_argument("--budget", type=float, default=12)
    ap.add_argument("--nar-query-chunk", type=int, default=1024)
    ap.add_argument("--backend", default="torch-eager", choices=("torch-eager", "torch"))
    ap.add_argument("--offload-ar", action="store_true")
    args = ap.parse_args()
    manifest = checked_manifest(args.experiment)
    variant = manifest["variants"][args.variant]
    import yue2.nar as nar
    from yue2 import YuE2Pipeline

    original = nar.synthesize

    def chunked(*a, **kw):
        kw.setdefault("query_chunk_size", args.nar_query_chunk)
        return original(*a, **kw)

    nar.synthesize = chunked
    started = time.monotonic()
    pipe = YuE2Pipeline(str(ROOT / "models" / "yue2" / "YuE2-3B"),
                       str(ROOT / "models" / "yue2" / "YuE2-Vae"),
                       device="cuda", memory_budget_gib=args.budget, backend=args.backend,
                       offload_ar=args.offload_ar, verify_hashes=False)
    song = pipe(**generation_kwargs(args.experiment, manifest, args.variant))
    directory = args.experiment / "renders" / args.variant
    directory.mkdir(parents=True, exist_ok=True)
    song.save(str(directory / "audio.wav"))
    song.save_artifacts(str(directory / "artifacts"))
    write_json(directory / "result.json", {"experiment_identity": manifest["experiment_identity"],
                                          "variant": args.variant, "seed": variant["seed"],
                                          "rendering": rendering_config(manifest),
                                          "truncated": song.truncated, "timing": song.timing,
                                          "elapsed_s": round(time.monotonic() - started, 2)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
