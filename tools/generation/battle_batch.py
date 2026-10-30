"""Battle-music batch: multi-candidate generation, ranked by arc + diction.

Same philosophy as the 20-song batch (source-clean, multi-candidate, a human
picks the take) with one addition that matters for this genre: because the user
asked for 欲扬先抑 -- restrained opening, late epic climax -- every candidate is
scored on its dynamic ARC (tools/generation/arc_qa.py) and on how much of the
intended lyric actually came back intelligible (tools/generation/diction_qa.py,
faster-whisper). The best-scoring take per title is promoted to
`selected/`, everything else is kept under `candidates/` for the human.

Ranking (deliberately simple and inspectable):
    arc   = verdict_bonus*3 + lift_db/6 + peak_at*2
            verdict_bonus: build-up=2, mild build=1, otherwise 0
    shape = -body_db/10                median level below the peak: real contrast
          + -mid_dip_db/10            the deepest sustained drop inside the song
          + onset_mean*4              percussive drive (the user keeps the
                                      rhythmic tracks)
          + low_share_peak*4          sub-bass weight at the climax
            (body_db and mid_dip_db are negative dB, so they enter negated)
    dict  = coverage*2 + ratio
    total = arc + shape + dict  (a take missing most of the words is warned)
Anything that fails the objective audio checks (duration/peak/clipping/silence)
is dropped before scoring.

`shape` was added 2026-10-15. The first three terms had already been maximised
by the 2026-10-14 batch, which the user then called flat: it had lift 9-19 dB and
a late climax but a median level only 2.7-6.8 dB under its own peak, so it
sounded loud throughout. These four terms are the ones that delivery did *not*
score well on.

Resumable: a title with a promoted take is skipped on the next run.

Usage:
  python tools/generation/battle_batch.py --inputs inputs/battle_batch \
      --out-dir outputs/battle_batch --seconds 120 --candidates 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc_qa  # noqa: E402
import clean_pipeline as cp  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASR_PYTHON = PROJECT_ROOT / "tools" / "asr" / ".venv" / "Scripts" / "python.exe"
DICTION_QA = PROJECT_ROOT / "tools" / "generation" / "diction_qa.py"

# Per-candidate sampling diversity. CFG stays at the measured best (2.0); the
# spread comes from temperature/top-k/seed so every take is guided equally hard.
CANDIDATE_PARAMS = [
    {"temperature": 0.80, "topk": 30, "seed_offset": 0},
    {"temperature": 1.00, "topk": 50, "seed_offset": 7},
    {"temperature": 1.10, "topk": 60, "seed_offset": 17},
    {"temperature": 0.90, "topk": 40, "seed_offset": 23},
]

VERDICT_BONUS = {"build-up": 2, "mild build": 1, "early peak": 0, "flat (no restraint)": 0}


def score_take(audio: dict, arc: dict, dic: dict | None,
               melody: dict | None = None) -> dict:
    arc_score = (VERDICT_BONUS.get(arc.get("verdict", ""), 0) * 3
                 + max(arc.get("lift_db", 0.0), 0.0) / 6.0
                 + arc.get("peak_at", 0.0) * 2.0)
    # Shape terms (2026-10-15). Every term is "more dynamic / harder hitting is
    # better": body_db and mid_dip_db are negative dB under the peak, so they
    # enter negated, and both are capped so one extreme take cannot carry the
    # ranking on its own.
    shape_score = (
        min(-float(arc.get("body_db", 0.0)), 15.0) / 10.0
        + min(-float(arc.get("mid_dip_db", 0.0)), 30.0) / 10.0
        + float(arc.get("onset_mean", 0.0)) * 4.0
        + float(arc.get("low_share_peak", 0.0)) * 4.0
    )
    if dic:
        dict_score = dic["coverage"] * 2.0 + dic["ratio"]
    else:
        dict_score = 0.0
    # Symbolic melody score (2026-10-26): the plan is available before render
    # for YuE2, and it is the only place where chorus/verse density, register
    # lift and hook recurrence can be measured without listening.  Keep it as a
    # modest term (0-10 points) so it breaks ties without outvoting the audio.
    melody_score = float(melody.get("score", 0.0)) / 10.0 if melody else 0.0
    return {
        "arc_score": round(arc_score, 3),
        "shape_score": round(shape_score, 3),
        "diction_score": round(dict_score, 3),
        "melody_score": round(melody_score, 3),
        "total": round(arc_score + shape_score + dict_score + melody_score, 3),
    }


def transcribe(takes: list[Path], lyrics: Path, language: str = "zh") -> dict[str, dict]:
    """One whisper load for all takes of a title (the model load dominates).

    `language` is "auto" for mixed zh/en material -- whisper then detects per
    file, which is the only sane option when one song switches language between
    sections.
    """
    if not takes or not ASR_PYTHON.exists():
        return {}
    out_json = takes[0].parent / "_diction.json"
    cmd = [str(ASR_PYTHON), "-I", str(DICTION_QA), "--lyrics", str(lyrics),
           "--language", language, "--json", str(out_json), *[str(t) for t in takes]]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not out_json.exists():
        print(f"    diction check skipped: {(proc.stderr or '')[-200:]}", flush=True)
        return {}
    data = json.loads(out_json.read_text(encoding="utf-8"))
    return {str(Path(d["file"]).name): d for d in data}


def load_langs(inputs: Path) -> dict[str, str]:
    """Optional slug -> language map (`_langs.json`).

    "zh" (default), "en", or "auto" for songs that mix languages -- whisper then
    detects per file instead of being forced, which is the only workable option
    when a verse is Chinese and the chorus is English.
    """
    path = inputs / "_langs.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def title_files(inputs: Path) -> list[tuple[str, Path, Path]]:
    """Every lyrics file with a matching tags file, sorted by name."""
    out = []
    for lyrics in sorted(inputs.glob("*_lyrics.txt")):
        slug = lyrics.name[: -len("_lyrics.txt")]
        tags = inputs / f"{slug}_tags.txt"
        if tags.exists():
            out.append((slug, lyrics, tags))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=PROJECT_ROOT / "inputs" / "battle_batch")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "battle_batch")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument("--base-cfg", type=float, default=2.0)
    ap.add_argument("--base-seed", type=int, default=500)
    ap.add_argument("--no-diction", action="store_true")
    ap.add_argument("--language", default="zh",
                    help="default lyric language for the diction check; "
                         "override per title in _langs.json")
    args = ap.parse_args()

    cand_dir = args.out_dir / "candidates"
    sel_dir = args.out_dir / "selected"
    cand_dir.mkdir(parents=True, exist_ok=True)
    sel_dir.mkdir(parents=True, exist_ok=True)

    titles = title_files(args.inputs)
    if not titles:
        print(f"no *_lyrics.txt/_tags.txt pairs in {args.inputs}")
        return 1
    langs = load_langs(args.inputs)
    print(f"{len(titles)} titles, {args.candidates} candidates each, "
          f"{args.seconds}s, cfg={args.base_cfg}", flush=True)
    if langs:
        print(f"languages: {langs}", flush=True)

    index_rows: list[dict] = []
    for title, lyrics, tags in titles:
        promoted = sel_dir / f"{title}.wav"
        if promoted.exists():
            print(f"[{time.strftime('%H:%M:%S')}] skip {title} (already promoted)", flush=True)
            continue

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = cand_dir / f"{title}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{time.strftime('%H:%M:%S')}] === {title} -> {run_dir.name}", flush=True)

        takes: list[dict] = []
        for i in range(min(args.candidates, len(CANDIDATE_PARAMS))):
            p = CANDIDATE_PARAMS[i]
            wav = run_dir / f"{title}_c{i + 1}.wav"
            seed = args.base_seed + p["seed_offset"]
            print(f"  [{time.strftime('%H:%M:%S')}] c{i + 1} seed={seed} "
                  f"temp={p['temperature']} topk={p['topk']}", flush=True)
            t0 = time.time()
            try:
                meta = cp.generate_one(lyrics, tags, wav, args.seconds, seed,
                                       p["temperature"], p["topk"], args.base_cfg)
            except Exception as exc:
                print(f"    FAILED: {str(exc)[-300:]}", flush=True)
                continue
            check = cp.check_audio(wav, min_duration_s=args.seconds * 0.6)
            takes.append({"wav": wav, "seed": seed, "params": p, "gen": meta,
                          "check": check, "gen_seconds": round(time.time() - t0, 1)})
            status = "OK" if check["ok"] else f"REJECT {check['reasons']}"
            print(f"    {status}", flush=True)

        good = [t for t in takes if t["check"]["ok"]]
        if not good:
            print(f"  no usable take for {title}", flush=True)
            continue

        dic = ({} if args.no_diction
               else transcribe([t["wav"] for t in good], lyrics,
                               langs.get(title, args.language)))
        for t in good:
            t["arc"] = arc_qa.analyze(t["wav"])
            t["diction"] = dic.get(t["wav"].name)
            t["score"] = score_take(t["check"], t["arc"], t["diction"])

        best = max(good, key=lambda t: t["score"]["total"])
        # Source-clean reference first, then a linear loudness-normalized
        # delivery copy (normalization does not change the spectrum).
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
        print(f"  -> {best['wav'].name}  arc={row['arc']['verdict']} "
              f"lift={row['arc']['lift_db']} peak@{row['arc']['peak_at']} "
              f"coverage={row['diction']}  total={best['score']['total']}", flush=True)

    if index_rows:
        idx = args.out_dir / "index.json"
        idx.write_text(json.dumps(index_rows, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"\nwrote {idx}")
    print(f"[{time.strftime('%H:%M:%S')}] batch done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
