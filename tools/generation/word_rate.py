"""How fast is the singer actually articulating, second by second?

Why this exists (2026-10-16): the user's complaint was that our songs keep one
rhythm from start to finish, while a reference like Wildfire is slow and
word-sparse in the verse and dense in the chorus. `plan_metrics.py` measures
that on our side (the symbolic plan). This tool measures it on *any* audio --
our rendered take, or a reference track that has no plan at all -- so the two
can be compared and the plan edited to match.

Method: faster-whisper word-level timestamps (the same model `diction_qa.py`
uses; already local at PR_AIMusic_Extension/models/faster-whisper-small).
Words per second over a sliding window is a good proxy for articulation rate --
one term per sung word/character regardless of how many notes it is held over.

Two modes:

  --window N   the density curve: words/s in each N-second window. Use this on
               a reference track; the shape of the curve is the target.
  --plan FILE  per-section words/s, with the section boundaries *derived from
               the ABC plan's own bars and tempo* and printed next to the plan's
               onset rate. This is the cross-check that the plan's density is
               really what comes out of the singer (verified 2026-10-16 on
               01_hammer_and_hymn: the two columns track section by section).

Language matters for the unit of "word": pass --language en for English
references, zh for Mandarin (whisper then segments by characters/phrases).
Mixed songs need --language auto, and the numbers are then only comparable
within the same song.

Runtime: cpu/int8 small model, roughly 1-3x faster than realtime, so a 3-minute
reference takes a couple of minutes. That is why the plan-side metric exists --
use it for iteration, and this tool for the reference and for spot checks.

Usage (must run under tools/asr/.venv, which has faster-whisper):
  tools/asr/.venv/Scripts/python.exe -I tools/generation/word_rate.py \
      --wav human_refs/refs/wildfire.wav --language en --window 10
  tools/asr/.venv/Scripts/python.exe -I tools/generation/word_rate.py \
      --wav outputs/2026-10-15_epic_en/selected/01_hammer_and_hymn.wav \
      --plan outputs/2026-10-15_epic_en/candidates/01_*/01_*_c1_artifacts/score.abc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_metrics import measure_plan  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "PR_AIMusic_Extension" / "models" / "faster-whisper-small"

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def transcribe_words(wav: Path, model_dir: Path, language: str | None):
    """[(start, end, word)] for every word faster-whisper reports."""
    from faster_whisper import WhisperModel
    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(wav), language=language,
                                      beam_size=5, vad_filter=False,
                                      word_timestamps=True)
    words = [(w.start, w.end, (w.word or "").strip())
             for s in segments for w in (s.words or []) if (w.word or "").strip()]
    return words, info


def sections_from_plan(plan: Path) -> list[tuple[str, float, float, float]]:
    """[(section, t0, t1, plan_onsets_per_s)] from the plan's bars and tempo."""
    rows = measure_plan(plan.read_text(encoding="utf-8-sig"))
    out: list[tuple[str, float, float, float]] = []
    clock = 0.0
    for row in rows:
        out.append((row["section"], clock, clock + row["seconds"],
                    row["onsets_per_s"]))
        clock += row["seconds"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--window", type=float, default=10.0,
                    help="density-curve window in seconds (0 disables the curve)")
    ap.add_argument("--plan", type=Path, default=None,
                    help="score.abc: also report per-section words/s using the "
                         "plan's own section boundaries")
    ap.add_argument("--language", default="en",
                    help="ISO code, or 'auto' (mixed-language songs)")
    ap.add_argument("--model", type=Path, default=MODEL_DIR)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    language = None if args.language in ("auto", "") else args.language
    words, info = transcribe_words(args.wav, args.model, language)
    if not words:
        print(f"{args.wav.name}: no words detected")
        return 1
    total = words[-1][1]
    print(f"{args.wav.name}: {len(words)} words, last ends {total:.1f}s, "
          f"lang {info.language} (p={info.language_probability:.2f})")
    print(f"whole track: {len(words) / total:.2f} words/s")

    payload: dict = {"file": str(args.wav), "words": len(words),
                     "lang": info.language, "end_s": round(total, 1),
                     "words_per_s": round(len(words) / total, 3)}

    if args.window and args.window > 0:
        print(f"\ndensity curve ({args.window:g}s windows)")
        curve = []
        t = 0.0
        while t < total:
            n = sum(1 for s, _, _ in words if t <= s < t + args.window)
            span = min(args.window, total - t)
            curve.append({"t0": round(t, 1), "words": n,
                          "words_per_s": round(n / span, 2) if span > 0 else 0.0})
            bar = "#" * int(round((n / span) * 10)) if span > 0 else ""
            print(f"{t:>7.1f}s {n:>4}  {n / span:>5.2f} w/s  {bar}")
            t += args.window
        payload["curve"] = curve

    if args.plan:
        print(f"\nper section (boundaries from {args.plan.name})")
        print(f"{'section':<14}{'t0':>8}{'t1':>8}{'plan ons/s':>12}"
              f"{'words':>7}{'word/s':>8}{'ratio':>7}")
        sections = {}
        for name, a, b, ons in sections_from_plan(args.plan):
            n = sum(1 for s, _, _ in words if a <= s < b)
            rate = n / (b - a) if b > a else 0.0
            sections.setdefault(name, []).append(
                {"t0": round(a, 1), "t1": round(b, 1), "words": n,
                 "words_per_s": round(rate, 2), "plan_onsets_per_s": ons})
            print(f"{name:<14}{a:>8.1f}{b:>8.1f}{ons:>12.2f}{n:>7}{rate:>7.2f}"
                  f"{rate / ons if ons else 0:>7.2f}")
        payload["sections"] = sections
        by_type = {}
        for name, group in sections.items():
            by_type[name] = round(sum(g["words_per_s"] for g in group) / len(group), 2)
        payload["section_mean_words_per_s"] = by_type
        if "verse" in by_type and "chorus" in by_type and by_type["verse"]:
            payload["chorus_over_verse"] = round(by_type["chorus"] / by_type["verse"], 2)
            print(f"\nchorus/verse density = {payload['chorus_over_verse']:.2f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\njson -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
