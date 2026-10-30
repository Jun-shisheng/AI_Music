from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import wave

from .workflow import YUE_PYTHON, dispatch, prepare, report, write_json


def init_annotations(reference: Path, out: Path, min_seconds=16.0):
    """Acoustic novelty draft: changes in local energy, bands and flux.

    No verse/chorus guesses are made. A listener must label/adjust boundaries.
    """
    from .audio import decode, descriptors, probe
    import numpy as np

    info = probe(reference)
    sr, window = 22050, 4.0
    samples = decode(reference, sr)
    rows = []
    for start in np.arange(0, info["duration_s"], window):
        y = samples[int(start*sr):int((start+window)*sr)]
        if len(y) < sr:
            break
        row = descriptors(y, sr)
        rows.append((float(start), [row["rms_dbfs"], *row["band_energy_share"].values(), row["mix_transient_events_per_s"]]))
    features = np.asarray([r[1] for r in rows])
    if len(features) >= 3:
        normalized = (features - np.median(features, axis=0)) / np.maximum(np.std(features, axis=0), .01)
        novelty = np.linalg.norm(np.diff(normalized, axis=0), axis=1)
        candidates = [i + 1 for i in range(1, len(novelty)-1)
                      if novelty[i] >= novelty[i-1] and novelty[i] >= novelty[i+1]
                      and novelty[i] >= np.percentile(novelty, 55)]
        chosen = []
        for index in sorted(candidates, key=lambda i: -novelty[i-1]):
            value = rows[index][0]
            if min(value, info["duration_s"]-value) >= min_seconds and all(abs(value-t) >= min_seconds for t in chosen):
                chosen.append(value)
        boundaries = [0.0, *sorted(chosen), info["duration_s"]]
    else:
        boundaries = [0.0, info["duration_s"]]
    data = {"schema": 1, "reference_audio": str(reference.resolve()), "status": "draft_acoustic_boundaries_unreviewed",
            "method": "4s feature novelty, local maxima, minimum segment spacing; boundaries approximate to +/-4s",
            "tempo_bpm": None, "first_beat_s": None, "beats_per_bar": 4,
            "instructions": "Listen, correct times, replace section_XX with verse/chorus/prechorus etc., annotate roles and mark reviewed. Do not infer instruments from bands.",
            "sections": [{"name": f"section_{i+1:02d}", "start_s": round(a, 3), "end_s": round(b, 3),
                          "status": "unreviewed", "roles": {}, "target": {"proposal_status": "unreviewed"}}
                         for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]))]}
    write_json(out, data)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local reference arrangement studio; no automatic perceptual quality claims")
    sub = ap.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-annotations", help="Acoustic boundary draft for human correction")
    init.add_argument("--reference", type=Path, required=True)
    init.add_argument("--out", type=Path, required=True)
    init.add_argument("--min-seconds", type=float, default=16)
    bp = sub.add_parser("blueprint")
    bp.add_argument("--reference", type=Path, required=True)
    bp.add_argument("--annotations", type=Path, required=True)
    bp.add_argument("--stems", type=Path, help="Optional JSON {role: aligned_audio_path}; relative paths resolve beside annotations")
    bp.add_argument("--out", type=Path, required=True)
    mapping = sub.add_parser("propose-mapping", help="Propose editable destination slots from measured contrast; source roles remain unknown")
    mapping.add_argument("--blueprint", type=Path, required=True)
    mapping.add_argument("--out", type=Path, required=True)
    mapping.add_argument("--target-sections", nargs=2, default=["verse", "chorus"])
    mapping.add_argument("--source-sections", nargs=2)
    mapping.add_argument("--replace-mapping", action="store_true")
    prep = sub.add_parser("prepare")
    prep.add_argument("--plan", type=Path, required=True)
    prep.add_argument("--lyrics", type=Path, required=True)
    prep.add_argument("--style", type=Path, required=True)
    prep.add_argument("--blueprint", type=Path)
    prep.add_argument("--out-dir", type=Path, required=True)
    prep.add_argument("--start-section", default="verse")
    prep.add_argument("--section-count", type=int, default=2)
    prep.add_argument("--sections", nargs="+", help="Consecutive selectors, e.g. verse:1 chorus:1")
    prep.add_argument("--excerpt-lyrics", type=Path)
    prep.add_argument("--seed", type=int, default=1338)
    prep.add_argument("--allow-long", action="store_true")
    prep.add_argument("--arranged-pattern", default="arpeggio", choices=("arpeggio", "support"),
                      help="support uses one/two chord-root/fifth notes per bar with half-bar rests; only changes arranged")
    prep.add_argument("--cot", default="full", choices=("full", "melody"))
    prep.add_argument("--cfg-scale", type=float, help="Optional YuE2 guidance scale; omit to keep native default")
    prep.add_argument("--semantic-max-tokens", type=int, help="Optional AR token/KV allocation ceiling; truncated outputs remain rejected")
    controlled = sub.add_parser("prepare-controlled", help="Prepare MIDI and independent track events on a fixed grid; no audio rendering")
    controlled.add_argument("--config", type=Path, required=True)
    controlled.add_argument("--vocal", type=Path, required=True)
    controlled.add_argument("--out-dir", type=Path, required=True)
    render = sub.add_parser("render")
    render.add_argument("--experiment", type=Path, required=True)
    render.add_argument("--variant", action="append", choices=("baseline", "sparse", "full", "arranged"))
    render.add_argument("--python", type=Path, default=YUE_PYTHON)
    render.add_argument("--budget", type=float, default=12)
    render.add_argument("--nar-query-chunk", type=int, default=1024)
    render.add_argument("--backend", default="torch-eager", choices=("torch-eager", "torch"))
    render.add_argument("--no-offload-ar", action="store_true")
    render.add_argument("--dry-run", action="store_true")
    render.add_argument("--force", action="store_true")
    render.add_argument("--timeout-minutes", type=float, default=40)
    reuse = sub.add_parser("reuse-renders", help="Explicitly reuse completed inference with byte-identical model inputs and runtime")
    reuse.add_argument("--from-experiment", type=Path, required=True)
    reuse.add_argument("--experiment", type=Path, required=True)
    reuse.add_argument("--variant", action="append", required=True, choices=("baseline", "sparse", "full", "arranged"))
    reuse.add_argument("--python", type=Path, default=YUE_PYTHON)
    reuse.add_argument("--budget", type=float, default=12)
    reuse.add_argument("--nar-query-chunk", type=int, default=1024)
    reuse.add_argument("--backend", default="torch-eager", choices=("torch-eager", "torch"))
    reuse.add_argument("--no-offload-ar", action="store_true")
    rep = sub.add_parser("report")
    rep.add_argument("--experiment", type=Path, required=True)
    rep.add_argument("--target-dbfs", type=float, default=-20)
    shape = sub.add_parser("shape", help="Deterministic plan-level density/dynamics shaping")
    shape.add_argument("--plan", type=Path, required=True)
    shape.add_argument("--out", type=Path, required=True)
    shape.add_argument("--reduce", action="append", default=[])
    shape.add_argument("--thin", action="append", default=[])
    shape.add_argument("--split", action="append", default=[])
    shape.add_argument("--hook", action="append", default=[])
    shape.add_argument("--report", action="store_true")
    midi = sub.add_parser("midi", help="Render a section-level MIDI arrangement and mix a vocal")
    midi.add_argument("--midi", type=Path, required=True)
    midi.add_argument("--vocal", type=Path, required=True)
    midi.add_argument("--out-dir", type=Path, required=True)
    midi.add_argument("--name", default=None)
    midi.add_argument("--mix-rate", type=int, default=44100, choices=[44100, 48000])
    midi.add_argument("--target-rms", type=float, default=0.07)
    args = ap.parse_args(argv)
    try:
        if args.command == "init-annotations":
            result = init_annotations(args.reference, args.out, args.min_seconds)
            summary = {"annotations": str(args.out.resolve()), "status": result["status"], "sections": len(result["sections"])}
        elif args.command == "blueprint":
            from .audio import blueprint
            stems = json.loads(args.stems.read_text(encoding="utf-8-sig")) if args.stems else None
            result = blueprint(args.reference, args.annotations, stems)
            write_json(args.out, result)
            summary = {"blueprint": str(args.out.resolve()), "annotation_status": result["annotation_status"], "sections": len(result["sections"])}
        elif args.command == "prepare":
            result = prepare(args.plan, args.lyrics, args.style, args.out_dir, args.blueprint,
                             args.start_section, args.section_count, args.sections, args.excerpt_lyrics,
                             args.seed, args.allow_long, args.arranged_pattern, args.cot, args.cfg_scale,
                             args.semantic_max_tokens)
            summary = {"experiment": str(args.out_dir.resolve()), "duration_s": result["excerpt"]["duration_s"],
                       "variants": list(result["variants"]), "claim_status": result["claim_status"],
                       "arrangement_mapping": result.get("arrangement_mapping", {"provenance": "legacy_unreported_requires_review"})}
        elif args.command == "prepare-controlled":
            from .controlled_arrangement import prepare_controlled
            result = prepare_controlled(args.config, args.vocal, args.out_dir)
            summary = {"experiment": str(args.out_dir.resolve()), "status": result["status"],
                       "kind": result["kind"], "variants": list(result["variants"]),
                       "audio_rendered": False}
        elif args.command == "propose-mapping":
            from .mapping import propose_mapping
            if args.blueprint.resolve() == args.out.resolve():
                raise ValueError("Use a new --out path to preserve the source blueprint")
            source = json.loads(args.blueprint.read_text(encoding="utf-8-sig"))
            result = propose_mapping(source, args.target_sections, args.source_sections, args.replace_mapping)
            write_json(args.out, result)
            summary = {"blueprint": str(args.out.resolve()), "mapping_proposal": result["mapping_proposal"]}
        elif args.command == "render":
            summary = dispatch(args.experiment, args.variant, args.python, args.budget,
                               args.nar_query_chunk, args.backend, not args.no_offload_ar,
                               args.dry_run, args.force, args.timeout_minutes)
        elif args.command == "shape":
            root = Path(__file__).resolve().parents[2]
            command = [sys.executable, str(root / "tools/generation/plan_shape.py"),
                       "--file", str(args.plan), "--out", str(args.out)]
            for option in ("reduce", "thin", "split", "hook"):
                for value in getattr(args, option):
                    command += [f"--{option}", value]
            if args.report:
                command.append("--report")
            subprocess.run(command, check=True)
            summary = {"plan": str(args.out.resolve()), "status": "shaped_not_rendered",
                       "note": "Only V: Vocal density/dynamics changed; render separately."}
        elif args.command == "midi":
            root = Path(__file__).resolve().parents[2]
            command = [sys.executable, str(root / "tools/generation/render_midi_arrangement.py"),
                       "--midi", str(args.midi), "--vocal", str(args.vocal),
                       "--out-dir", str(args.out_dir), "--mix-rate", str(args.mix_rate),
                       "--target-rms", str(args.target_rms)]
            if args.name:
                command += ["--name", args.name]
            subprocess.run(command, check=True)
            summary = {"out_dir": str(args.out_dir.resolve()),
                       "status": "rendered_awaiting_listening",
                       "alignment_status": "unverified_requires_listening"}
        elif args.command == "reuse-renders":
            from .reuse import reuse_renders
            summary = reuse_renders(args.from_experiment, args.experiment, args.variant, args.python,
                                    args.budget, args.nar_query_chunk, args.backend, not args.no_offload_ar)
        else:
            result = report(args.experiment, args.target_dbfs)
            summary = {"report": str((args.experiment / "REPORT.md").resolve()), "status": result["status"],
                       "render_states": result["render_states"], "quality_claim": result["quality_claim"]}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 1 if summary.get("status") == "failed" else 0
    except (ValueError, OSError, KeyError, wave.Error) as exc:
        print(f"reference_studio: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
