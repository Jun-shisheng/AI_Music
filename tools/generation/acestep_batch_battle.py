"""ACE-Step battle pass: the same briefs, for an engine A/B.

The HeartMuLa batch is the primary (it actually sings the intended lyric --
measured coverage 0.91 vs ACE-Step's 0.52). This script renders the *same*
lyrics and descriptors through ACE-Step so the human can compare engines on
identical material: ACE is much faster, licenses cleaner (MIT) and is less
harsh in the 8-16 kHz band, but drops lyric sections.

Scoring/ranking is imported from battle_batch so both passes are judged by the
same rule; the models are loaded once for the whole run (ACE generation is
seconds per take, so reloading would dominate).

Usage:
  python tools/generation/acestep_batch_battle.py --inputs inputs/battle_batch \
      --out-dir outputs/battle_batch_ace --seconds 120 --candidates 2 --offload
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACE_ROOT = PROJECT_ROOT / "vendor" / "ACE-Step-1.5"
sys.path.insert(0, str(ACE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc_qa  # noqa: E402
import battle_batch as bb  # noqa: E402
import clean_pipeline as cp  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import (  # noqa: E402
    GenerationConfig,
    GenerationParams,
    generate_music,
)
from acestep.llm_inference import LLMHandler  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=PROJECT_ROOT / "inputs" / "battle_batch")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "outputs" / "battle_batch_ace")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument("--dit", default="acestep-v15-turbo")
    ap.add_argument("--lm", default="acestep-5Hz-lm-0.6B")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--no-diction", action="store_true")
    ap.add_argument("--language", default="zh",
                    help="default lyric language for the diction check; "
                         "override per title in _langs.json")
    args = ap.parse_args()

    cand_root = args.out_dir / "candidates"
    sel_dir = args.out_dir / "selected"
    cand_root.mkdir(parents=True, exist_ok=True)
    sel_dir.mkdir(parents=True, exist_ok=True)

    titles = bb.title_files(args.inputs)
    if not titles:
        print(f"no briefs in {args.inputs}")
        return 1
    langs = bb.load_langs(args.inputs)
    if langs:
        print(f"languages: {langs}", flush=True)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading ACE-Step {args.dit} ...", flush=True)
    dit = AceStepHandler()
    msg, ok = dit.initialize_service(
        project_root=str(ACE_ROOT), config_path=args.dit, device="cuda",
        offload_to_cpu=args.offload, offload_dit_to_cpu=args.offload)
    if not ok:
        print(f"dit init failed: {msg}")
        return 1
    llm = LLMHandler()
    msg, ok = llm.initialize(checkpoint_dir=str(ACE_ROOT / "checkpoints"),
                             lm_model_path=args.lm, backend="pt", device="cuda",
                             offload_to_cpu=args.offload)
    if not ok:
        print(f"lm init failed: {msg}")
        return 1
    print(f"[{time.strftime('%H:%M:%S')}] models ready in {time.time() - t0:.0f}s", flush=True)

    index_rows: list[dict] = []
    for title, lyrics_path, tags_path in titles:
        promoted = sel_dir / f"{title}.wav"
        if promoted.exists():
            print(f"[{time.strftime('%H:%M:%S')}] skip {title} (already promoted)", flush=True)
            continue

        run_dir = cand_root / f"{title}_{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # The HeartMuLa tag file doubles as the ACE caption: it is already the
        # full style description for the same track.
        caption = " ".join(tags_path.read_text(encoding="utf-8").split())
        lyrics = lyrics_path.read_text(encoding="utf-8")

        params = GenerationParams(
            caption=caption, lyrics=lyrics, instrumental=False,
            vocal_language="zh", duration=args.seconds, thinking=True,
            inference_steps=8, shift=3.0,
        )
        config = GenerationConfig(batch_size=args.candidates, audio_format="wav",
                                  use_random_seed=True)
        print(f"[{time.strftime('%H:%M:%S')}] === {title} x{args.candidates}", flush=True)
        tg = time.time()
        result = generate_music(dit, llm, params, config, save_dir=str(run_dir))
        if not result.success:
            print(f"    FAILED: {result.error}", flush=True)
            continue

        takes: list[dict] = []
        for i, audio in enumerate(result.audios, start=1):
            src = Path(audio["path"])
            wav = run_dir / f"{title}_c{i}.wav"
            if src.resolve() != wav.resolve():
                shutil.copy2(src, wav)
            check = cp.check_audio(wav, min_duration_s=args.seconds * 0.6)
            takes.append({"wav": wav, "seed": (audio.get("params") or {}).get("seed"),
                          "check": check})
            if not check["ok"]:
                print(f"    c{i} REJECT {check['reasons']}", flush=True)
        good = [t for t in takes if t["check"]["ok"]]
        if not good:
            continue
        print(f"    {len(good)} usable takes in {time.time() - tg:.0f}s", flush=True)

        dic = ({} if args.no_diction
               else bb.transcribe([t["wav"] for t in good], lyrics_path,
                                  langs.get(title, args.language)))
        for t in good:
            t["arc"] = arc_qa.analyze(t["wav"])
            t["diction"] = dic.get(t["wav"].name)
            t["score"] = bb.score_take(t["check"], t["arc"], t["diction"])
        best = max(good, key=lambda t: t["score"]["total"])

        shutil.copy2(best["wav"], sel_dir / f"{title}_raw.wav")
        norm = cp.normalize_loudness(best["wav"], promoted)
        row = {
            "title": title,
            "selected": best["wav"].name,
            "seed": best["seed"],
            "score": best["score"],
            "arc": {k: v for k, v in best["arc"].items() if k != "envelope"},
            "envelope": best["arc"]["envelope"],
            "diction": (best["diction"] or {}).get("coverage"),
            "raw": f"{title}_raw.wav",
            "delivery": promoted.name,
            "gain_db": norm.get("gain_db"),
            "candidates": [
                {"file": t["wav"].name, "seed": t["seed"],
                 "verdict": t["arc"]["verdict"], "lift_db": t["arc"]["lift_db"],
                 "peak_at": t["arc"]["peak_at"],
                 "coverage": (t["diction"] or {}).get("coverage"),
                 "score": t["score"]["total"]}
                for t in good
            ],
        }
        index_rows.append(row)
        (run_dir / "result.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {best['wav'].name} arc={row['arc']['verdict']} "
              f"lift={row['arc']['lift_db']} peak@{row['arc']['peak_at']} "
              f"coverage={row['diction']} total={best['score']['total']}", flush=True)

    if index_rows:
        idx = args.out_dir / "index.json"
        idx.write_text(json.dumps(index_rows, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"\nwrote {idx}")
    print(f"[{time.strftime('%H:%M:%S')}] ACE pass done "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
