"""YuE2 battle pass: the same briefs through the structurally stronger backend.

Why this exists: on the identical brief, YuE2 measured far better on exactly the
property the user asked for -- restrained opening, deep bridge, late and large
climax (lift 20.4 dB vs HeartMuLa's 11.3, quiet share 0.28 vs 0.12, a -26 dB
dip before the climax). It also plans the piece as symbolic ABC notation first,
which HeartMuLa has no equivalent of, and writes that plan out as an artifact.

It is slower (~11 min/take on this 12 GB card) and needs three accommodations
the package's own README assumes away (24 GB GPU, Linux, FlashAttention):

  * `--budget 12`          cap the process's CUDA allocation
  * `--backend torch-eager` no CUDA graphs (the graph path calls
                            aten._flash_attention_forward, which is a stub in
                            the Windows PyTorch build)
  * `--nar-query-chunk`     the NAR prefill otherwise runs one query block the
                            size of the whole song and OOMs; the pipeline's
                            high-level API never passes this even though
                            nar.synthesize supports it

The model is loaded once for the whole run. Output schema matches
battle_batch.py, so battle_review.py renders it without changes.

Usage:
  vendor/YuE2/.venv/Scripts/python.exe tools/generation/yue2_batch_battle.py \
      --inputs inputs/battle_batch --out-dir outputs/battle_batch_yue2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc_qa  # noqa: E402
import battle_batch as bb  # noqa: E402
import clean_pipeline as cp  # noqa: E402
import melody_qa  # noqa: E402
import plan_metrics  # noqa: E402
import plan_shape  # noqa: E402
import yue2_plan_edit  # noqa: E402


def _empty_cache() -> None:
    """Release cached CUDA blocks between the plan and render stages.

    2026-10-23: the plan pass leaves large freed blocks behind; the render pass
    then fails to find a contiguous one and OOMs even though the card has room
    ("under 200 MiB unallocated" in the error). `expandable_segments` is not
    supported on Windows, so returning the cache is the lever that is left.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def plan_from_file(path: Path) -> str:
    """Read a pre-shaped plan for `abc=` rendering, rejecting a file that is not.

    A plan that lost its `% section` markers or its `V: Vocal` voice would still
    render -- as something unrelated to what was measured. `plan_shape.py` and
    `plan_metrics.py` both key off those markers, so a malformed plan silently
    invalidates the whole measurement chain. Fail here instead.
    """
    abc = path.read_text(encoding="utf-8-sig")
    if not any(line.startswith("%") for line in abc.splitlines()):
        raise ValueError(f"{path} has no '% section' markers; not an ABC plan")
    if "V: Vocal" not in abc:
        raise ValueError(f"{path} has no 'V: Vocal' voice; not a singable plan")
    return abc


def shape_is_good(arc: dict, body_ceiling_db: float = -6.0) -> bool:
    """Is this take already the shape the user asked for?

    build-up arc, late climax, and a body that actually sits below its own peak
    (`body_db` <= -6 dB; the 2026-10-14 delivery ran -2.7 to -6.8 dB, which the
    user heard as flat).
    """
    return (arc.get("verdict") == "build-up"
            and float(arc.get("peak_at", 0.0)) >= 0.75
            and float(arc.get("body_db", 0.0)) <= body_ceiling_db)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path,
                    default=PROJECT_ROOT / "inputs" / "battle_batch")
    ap.add_argument("--inputs2", type=Path, default=None,
                    help="optional second brief folder (e.g. the bilingual set)")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "outputs" / "battle_batch_yue2")
    ap.add_argument("--candidates", type=int, default=1,
                    help="takes per title; YuE2 is ~11 min each, so 1 by default")
    ap.add_argument("--seed-base", type=int, default=900)
    ap.add_argument("--retries", type=int, default=0,
                    help="extra attempts per take with a new seed; a seed whose "
                         "sampled plan is too long OOMs in 3-4 min, so retrying "
                         "beats chasing the allocator")
    ap.add_argument("--budget", type=float, default=12.0)
    ap.add_argument("--nar-query-chunk", type=int, default=2048)
    ap.add_argument("--max-plan-seconds", type=float, default=240.0,
                    help="skip/retry a sampled plan longer than this; on a 12 GB "
                         "card a 4+ minute plan OOMs in NAR even with small chunks")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.0,
                    help="override torch.cuda.set_per_process_memory_fraction "
                         "after the pipeline is built (0 = keep the pipeline's "
                         "own budget-minus-2GiB value). 0.90 on a 12 GiB card "
                         "recovers the headroom that long plans need.")
    ap.add_argument("--backend", default="torch-eager")
    ap.add_argument("--no-offload-ar", action="store_true")
    ap.add_argument("--model", type=Path,
                    default=PROJECT_ROOT / "models" / "yue2" / "YuE2-3B")
    ap.add_argument("--vae", type=Path,
                    default=PROJECT_ROOT / "models" / "yue2" / "YuE2-Vae")
    ap.add_argument("--no-diction", action="store_true")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--only", default=None,
                    help="only titles whose slug contains this substring; use "
                         "one title per invocation after an OOM, so each gets a "
                         "fresh process and therefore a clean allocator")
    ap.add_argument("--early-accept", dest="early_accept", action="store_true",
                    default=True,
                    help="stop after the first take when it already has a "
                         "build-up arc, a late climax and real body contrast")
    ap.add_argument("--no-early-accept", dest="early_accept", action="store_false",
                    help="always render --candidates takes per title")
    ap.add_argument("--early-accept-body-db", type=float, default=-6.0,
                    help="body_db ceiling for the early-accept rule")
    ap.add_argument("--final-lift", type=int, default=0,
                    help="transpose the plan's final chorus up by this many "
                         "semitones before rendering (0 = off). The plans write "
                         "the final chorus as a literal repeat of the first one; "
                         "the reference songs modulate up a step there.")
    ap.add_argument("--hook-max-tile", type=float, default=0.0,
                    help="skip hook copying when the target chorus is more than "
                         "this many times the source bars (0 = no guard)")
    ap.add_argument("--melody-target", default="zh-pop",
                    help="measure each take's symbolic plan with melody_qa.py "
                         "and fold its score into the ranking (default zh-pop; "
                         "use 'diagnostic' to measure without thresholds)")
    ap.add_argument("--no-melody", action="store_true",
                    help="skip the symbolic melody audit")
    ap.add_argument("--shape", action="append", default=[],
                    help="plan_shape.py operator run on the sampled plan before "
                         "rendering, e.g. --shape 'split chorus:2' (repeatable). "
                         "Implies the plan-then-render path (one extra ~60 s plan "
                         "pass). 2026-10-16: the user's complaint that the whole "
                         "song runs at one rhythm; 'split chorus:2' measured a "
                         "chorus density of 1.16 -> 2.08 words/s without touching "
                         "the verse.")
    ap.add_argument("--plan-file", type=Path, default=None,
                    help="render from this ABC plan verbatim instead of sampling "
                         "one. The seed still selects the AR/NAR sampling, so the "
                         "same seed with a reshaped plan is a controlled A/B (used "
                         "by the 2026-10-16 density experiment, plan_shape.py).")
    ap.add_argument("--intro-bars", type=int, default=None,
                    help="trim the plan's instrumental intro to this many bars "
                         "before rendering (see yue2_plan_edit). Unset = render "
                         "the sampled plan as-is. The 2026-10-14 batch began "
                         "singing at 15-32 s; the user wants a few seconds.")
    ap.add_argument("--lora-ar", type=Path, default=None,
                    help="PEFT LoRA safetensors folded into the AR/plan branch "
                         "(e.g. models/yue2/loras/tsfh-steps-from-hell/adapter-ar-195)")
    ap.add_argument("--lora-nar", type=Path, default=None,
                    help="PEFT LoRA safetensors folded into the NAR/timbre branch")
    ap.add_argument("--ar-scale", type=float, default=1.0,
                    help="strength multiplier on the AR LoRA (adapter recommends "
                         "0.5 for tsfh-steps-from-hell)")
    ap.add_argument("--nar-scale", type=float, default=1.0,
                    help="strength multiplier on the NAR LoRA (adapter recommends "
                         "0.375 for tsfh-steps-from-hell)")
    args = ap.parse_args()

    from yue2 import YuE2Pipeline
    import yue2.nar as nar

    if args.nar_query_chunk:
        _original = nar.synthesize

        def _chunked(*a, **kw):
            kw.setdefault("query_chunk_size", args.nar_query_chunk)
            return _original(*a, **kw)

        nar.synthesize = _chunked

    sources = [args.inputs] + ([args.inputs2] if args.inputs2 else [])
    briefs: list[tuple[Path, str, Path, Path, str]] = []
    for src in sources:
        langs = bb.load_langs(src)
        for slug, lyrics, tags in bb.title_files(src):
            if args.only and args.only not in slug:
                continue
            briefs.append((src, slug, lyrics, tags, langs.get(slug, args.language)))
    if not briefs:
        print("no briefs found")
        return 1

    cand_root = args.out_dir / "candidates"
    sel_dir = args.out_dir / "selected"
    cand_root.mkdir(parents=True, exist_ok=True)
    sel_dir.mkdir(parents=True, exist_ok=True)

    # The index must accumulate across processes. run_epic_yue2.py runs one
    # process per title (the cumulative-allocator OOM), so a fresh `index_rows`
    # per process would leave index.json holding only the last title -- and
    # run_epic_yue2.read_first_vocal() reads it to apply the vocal-onset gate.
    index_path = args.out_dir / "index.json"
    index_rows: list[dict] = []
    if index_path.exists():
        try:
            index_rows = [r for r in json.loads(index_path.read_text(encoding="utf-8"))
                          if isinstance(r, dict)]
        except Exception:
            index_rows = []

    print(f"[{time.strftime('%H:%M:%S')}] loading YuE2 stream {args.budget} GiB, "
          f"backend={args.backend}, chunk={args.nar_query_chunk} ...", flush=True)
    t0 = time.time()
    pipe = YuE2Pipeline(
        str(args.model), str(args.vae), device="cuda",
        memory_budget_gib=args.budget, backend=args.backend,
        offload_ar=not args.no_offload_ar, verify_hashes=False)
    # The pipeline pins the process to (budget - 2) GiB. On a 12 GiB card that
    # is 10 GiB, which is the wall the long English plans hit on 2026-10-22
    # (the error says "under 200 MiB unallocated" with ~9.7 GiB in use). The
    # desktop holds ~0.9 GiB, so a fraction override recovers the headroom
    # without touching the vendor package.
    if args.gpu_mem_fraction:
        import torch

        # The device argument takes an index, not the string "cuda".
        torch.cuda.set_per_process_memory_fraction(
            args.gpu_mem_fraction, torch.cuda.current_device())
        print(f"    gpu memory fraction -> {args.gpu_mem_fraction}", flush=True)
    if args.lora_ar or args.lora_nar:
        from yue2_lora import merge_pair

        for report in merge_pair(pipe._load_model(), args.lora_ar, args.lora_nar,
                                 ar_scale=args.ar_scale, nar_scale=args.nar_scale):
            print(f"    lora {Path(report['file']).name}: merged {report['merged']} "
                  f"linears, missing {len(report['missing'])}, "
                  f"scales {report['scales']}", flush=True)
            if report["missing"]:
                print(f"    WARNING unresolved: {report['missing'][:5]}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] loaded in {time.time() - t0:.0f}s; "
          f"{len(briefs)} titles", flush=True)

    for src, slug, lyrics_path, tags_path, lang in briefs:
        promoted = sel_dir / f"{slug}.wav"
        if promoted.exists():
            print(f"[{time.strftime('%H:%M:%S')}] skip {slug} (already promoted)", flush=True)
            continue
        run_dir = cand_root / f"{slug}_{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        style = " ".join(tags_path.read_text(encoding="utf-8").split())
        lyrics = lyrics_path.read_text(encoding="utf-8")

        takes: list[dict] = []
        for i in range(args.candidates):
            wav = run_dir / f"{slug}_c{i + 1}.wav"
            # YuE2 samples the length of its symbolic plan, so one seed can plan
            # a piece that does not fit and blow up in the AR or NAR stage (the
            # failure lands in 3-4 min, and it is NOT allocator fragmentation --
            # the OOM message shows under 200 MiB unallocated). Retrying with a
            # different seed is far cheaper than fighting the allocator.
            for attempt in range(args.retries + 1):
                seed = args.seed_base + attempt * 101 + i * 37
                print(f"[{time.strftime('%H:%M:%S')}] {slug} c{i + 1} seed={seed} "
                      f"(attempt {attempt + 1}/{args.retries + 1}) ...", flush=True)
                tg = time.time()
                try:
                    plan_info: dict = {}
                    if args.plan_file:
                        edited = plan_from_file(args.plan_file)
                        plan_info = {"plan_file": str(args.plan_file),
                                     "plan_sha_ok": True}
                        print(f"    plan: rendering from {args.plan_file.name} "
                              f"({len(edited)} chars)", flush=True)
                        _empty_cache()
                        song = pipe(style=style, lyrics=lyrics, cot="full",
                                    seed=seed, abc=edited)
                    elif args.intro_bars is None and not args.shape:
                        _empty_cache()
                        song = pipe(style=style, lyrics=lyrics, cot="full", seed=seed)
                    else:
                        # Sample the plan, edit it (intro trim, then the shape
                        # operators), then render from the edited plan
                        # (request.abc skips the plan stage). Costs one extra plan
                        # pass (~60 s) and does not depend on the model choosing a
                        # short intro or a contrasting density by itself.
                        plan = pipe.plan(style=style, lyrics=lyrics, cot="full",
                                         seed=seed)
                        (run_dir / f"{slug}_c{i + 1}_plan_original.abc").write_text(
                            plan.abc, encoding="utf-8")
                        edited, plan_info = ((plan.abc, {}) if not args.intro_bars
                                             else yue2_plan_edit.trim_intro(
                                                 plan.abc, keep_bars=args.intro_bars))
                        if args.shape:
                            edited, shape_report = plan_shape.apply_ops(
                                edited, args.shape,
                                hook_max_tile=args.hook_max_tile or None)
                            plan_info["shape"] = shape_report
                            for entry in shape_report:
                                print(f"    shape: {entry['section']} "
                                      f"{entry['mode']} {entry['amount']:g} "
                                      f"-> {entry['bars']} bars", flush=True)
                            if not shape_report:
                                print("    shape: no section matched; plan "
                                      "unchanged", flush=True)
                        if args.final_lift:
                            edited, lift_info = yue2_plan_edit.transpose_section(
                                edited, section="chorus", semitones=args.final_lift)
                            plan_info["final_lift"] = lift_info
                            if lift_info.get("changed"):
                                print(f"    plan: final chorus "
                                      f"{lift_info['section']} "
                                      f"+{args.final_lift} semitones", flush=True)
                        plan_seconds = sum(float(row.get("seconds", 0.0))
                                           for row in plan_metrics.measure_plan(edited))
                        if plan_seconds > args.max_plan_seconds:
                            print(f"    SKIP: plan {plan_seconds:.1f}s exceeds "
                                  f"{args.max_plan_seconds:.0f}s; retrying with the next seed",
                                  flush=True)
                            continue
                        # Render from the edited plan whenever anything actually
                        # edited it: the intro trim reports `changed`, the shape
                        # operators report their own entries. Checking only
                        # `changed` would silently render the *unshaped* plan
                        # whenever the intro was already short.
                        if plan_info.get("changed") or args.shape:
                            if plan_info.get("changed"):
                                print(f"    plan: intro {plan_info['intro_bars_before']}"
                                      f" -> {plan_info['intro_bars_after']} bars "
                                      f"(-{plan_info['dropped_seconds']}s)", flush=True)
                            _empty_cache()
                            song = pipe(style=style, lyrics=lyrics, cot="full",
                                        seed=seed, abc=edited)
                        else:
                            _empty_cache()
                            song = pipe(style=style, lyrics=lyrics, cot="full",
                                        seed=seed)
                    song.save(str(wav))
                    # The ABC plan is the whole point of this backend: keep it.
                    art_dir = run_dir / f"{slug}_c{i + 1}_artifacts"
                    song.save_artifacts(str(art_dir))
                    melody = None
                    if not args.no_melody:
                        abc_path = art_dir / "score.abc"
                        if abc_path.exists():
                            try:
                                mr = melody_qa.audit(
                                    abc_path.read_text(encoding="utf-8-sig"),
                                    target=args.melody_target)
                                melody = asdict(mr)
                                (art_dir / "melody_qa.json").write_text(
                                    json.dumps(melody, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
                                print(f"    melody: {mr.summary()} "
                                      f"(ratio={mr.density_ratio} "
                                      f"lift={mr.register_lift} "
                                      f"hook={mr.hook_similarity})", flush=True)
                            except Exception as exc:
                                print(f"    melody check skipped: {str(exc)[-160:]}",
                                      flush=True)
                except Exception as exc:
                    print(f"    FAILED: {str(exc)[-200:]}", flush=True)
                    # Return the cache before the next seed: a failed large
                    # allocation leaves freed-but-cached blocks behind, and the
                    # retry inherits the fragmentation.
                    _empty_cache()
                    continue
                check = cp.check_audio(wav, min_duration_s=60.0)
                takes.append({"wav": wav, "seed": seed, "check": check,
                              "plan_edit": plan_info, "melody": melody,
                              "gen_seconds": round(time.time() - tg, 1)})
                print(f"    {'OK' if check['ok'] else 'REJECT ' + str(check['reasons'])} "
                      f"({time.time() - tg:.0f}s)", flush=True)
                break

            # Adaptive second take (2026-10-15). A take costs 11-31 min on this
            # card, so rendering a second one for a title whose first take
            # already has the shape we want is the most expensive way to decide
            # nothing. `--candidates` stays the ceiling; this is the stop rule.
            if (args.early_accept and i + 1 < args.candidates and takes
                    and takes[-1]["check"]["ok"]):
                arc_now = arc_qa.analyze(takes[-1]["wav"])
                takes[-1]["arc"] = arc_now
                if shape_is_good(arc_now, args.early_accept_body_db):
                    print(f"    first take already has the shape "
                          f"(verdict={arc_now['verdict']} peak@{arc_now['peak_at']} "
                          f"body={arc_now['body_db']}dB) -- keeping it", flush=True)
                    break

        good = [t for t in takes if t["check"]["ok"]]
        if not good:
            continue
        dic = ({} if args.no_diction
               else bb.transcribe([t["wav"] for t in good], lyrics_path, lang))
        for t in good:
            t["arc"] = arc_qa.analyze(t["wav"])
            t["diction"] = dic.get(t["wav"].name)
            t["score"] = bb.score_take(t["check"], t["arc"], t["diction"],
                                       t.get("melody"))
        best = max(good, key=lambda t: t["score"]["total"])

        shutil.copy2(best["wav"], sel_dir / f"{slug}_raw.wav")
        norm = cp.normalize_loudness(best["wav"], promoted)
        best_diction = best["diction"] or {}
        row = {
            "title": slug,
            "selected": best["wav"].name,
            "seed": best["seed"],
            "score": best["score"],
            "arc": {k: v for k, v in best["arc"].items() if k != "envelope"},
            "envelope": best["arc"]["envelope"],
            "diction": best_diction.get("coverage"),
            # Where the first sung word lands -- the user's "前奏太长" metric.
            "first_vocal_s": best_diction.get("first_vocal_s"),
            "last_vocal_s": best_diction.get("last_vocal_s"),
            "plan_edit": best.get("plan_edit") or {},
            "melody": ({k: best["melody"].get(k) for k in
                        ("passes", "score", "density_ratio", "register_lift",
                         "hook_similarity", "overall_range", "climax_position")}
                       if best.get("melody") else None),
            "raw": f"{slug}_raw.wav",
            "delivery": promoted.name,
            "gain_db": norm.get("gain_db"),
            "candidates": [
                {"file": t["wav"].name, "seed": t["seed"],
                 "verdict": t["arc"]["verdict"], "lift_db": t["arc"]["lift_db"],
                 "peak_at": t["arc"]["peak_at"],
                 "coverage": (t["diction"] or {}).get("coverage"),
                 "first_vocal_s": (t["diction"] or {}).get("first_vocal_s"),
                 "score": t["score"]["total"],
                 "melody_score": (t.get("melody") or {}).get("score"),
                 "melody_passes": (t.get("melody") or {}).get("passes")}
                for t in good
            ],
        }
        index_rows = [r for r in index_rows if r.get("title") != slug] + [row]
        (run_dir / "result.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {best['wav'].name} arc={row['arc']['verdict']} "
              f"lift={row['arc']['lift_db']} peak@{row['arc']['peak_at']} "
              f"coverage={row['diction']}", flush=True)

        # Merge into a cumulative index so a crash mid-run still reviews.
        (args.out_dir / "index.json").write_text(
            json.dumps(index_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{time.strftime('%H:%M:%S')}] YuE2 pass done "
          f"({(time.time() - t0) / 60:.1f} min, {len(index_rows)} titles)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
