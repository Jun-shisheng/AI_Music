"""Promote one specific candidate take to `selected/` and update `index.json`.

The batch's ranker is a good default, but not the final word: on 2026-10-26
06_lantern c1 scored slightly higher than c2 while c2 had much better lyric
intelligibility (coverage 1.11 vs 0.76).  A human/delivery decision needs to be
able to override the ranker without re-rendering.

This recomputes the same numbers the batch would have written (arc, diction,
melody, score), copies the take, normalizes it linearly, and rewrites the row.

Usage:
  P=vendor/ACE-Step-1.5/.venv/Scripts/python.exe
  $P -I tools/generation/promote_candidate.py \
      --out-dir outputs/2026-10-26_zh_pop \
      --inputs inputs/2026-10-26_zh_pop \
      --slug 06_lantern_on_the_pass --candidate c2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "generation"))

import arc_qa  # noqa: E402
import battle_batch as bb  # noqa: E402
import clean_pipeline as cp  # noqa: E402
import melody_qa  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--candidate", default="c2",
                    help="c1/c2 or the full candidate stem")
    ap.add_argument("--language", default="zh")
    args = ap.parse_args()

    slug = args.slug
    pattern = f"{slug}_{args.candidate}.wav"
    matches = sorted((args.out_dir / "candidates").glob(f"*/{pattern}"))
    if not matches:
        print(f"no candidate {pattern} under {args.out_dir/'candidates'}",
              file=sys.stderr)
        return 1
    wav = matches[-1]
    plan = wav.parent / f"{wav.stem}_artifacts" / "score.abc"
    lyrics = args.inputs / f"{slug}_lyrics.txt"
    if not plan.exists() or not lyrics.exists():
        print(f"missing plan or lyrics for {wav.name}", file=sys.stderr)
        return 1

    check = cp.check_audio(wav, min_duration_s=60.0)
    if not check.get("ok"):
        print(f"candidate fails audio check: {check.get('reasons')}", file=sys.stderr)
        return 1
    arc = arc_qa.analyze(wav)
    dic = bb.transcribe([wav], lyrics, args.language).get(wav.name)
    melody = asdict(melody_qa.audit(plan.read_text(encoding="utf-8-sig"),
                                    target="zh-pop"))
    score = bb.score_take(check, arc, dic, melody)

    sel = args.out_dir / "selected"
    sel.mkdir(parents=True, exist_ok=True)
    raw = sel / f"{slug}_raw.wav"
    shutil.copy2(wav, raw)
    delivery = sel / f"{slug}.wav"
    norm = cp.normalize_loudness(raw, delivery)

    index_path = args.out_dir / "index.json"
    rows = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    row = next((r for r in rows if r.get("title") == slug), None)
    if row is None:
        row = {"title": slug, "candidates": []}
        rows.append(row)
    candidate_seed = next((c.get("seed") for c in row.get("candidates", [])
                           if c.get("file") == wav.name), None)
    row.update({
        "selected": wav.name,
        "seed": candidate_seed,
        "score": score,
        "arc": {k: v for k, v in arc.items() if k != "envelope"},
        "envelope": arc.get("envelope"),
        "diction": (dic or {}).get("coverage"),
        "first_vocal_s": (dic or {}).get("first_vocal_s"),
        "last_vocal_s": (dic or {}).get("last_vocal_s"),
        "melody": {k: melody.get(k) for k in
                   ("passes", "score", "density_ratio", "register_lift",
                    "hook_similarity", "overall_range", "climax_position")},
        "raw": raw.name,
        "delivery": delivery.name,
        "gain_db": norm.get("gain_db"),
        "promoted_from": str(wav),
    })
    index_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"promoted {wav.name} -> {delivery.name}")
    print(f"  arc={arc['verdict']} lift={arc['lift_db']} peak@{arc['peak_at']} "
          f"coverage={(dic or {}).get('coverage')} melody={melody['score']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
