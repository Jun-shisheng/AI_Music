"""Experiment preparation, isolated render dispatch, and listening reports."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from . import plan

ROOT = Path(__file__).resolve().parents[2]
YUE_PYTHON = ROOT / "vendor" / "YuE2" / ".venv" / "Scripts" / "python.exe"
VARIANTS = ("baseline", "sparse", "full", "arranged")
SEMANTIC_DEFAULTS = {"temperature": 1.0, "top_p": .95, "top_k": 100,
                     "repetition_penalty": 1.2, "penalty_window": 50,
                     "min_tokens": 200, "max_tokens": 9000}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def request_fingerprint(experiment_identity: str, command: list[str]) -> str:
    runtime = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return plan.sha(json.dumps({"experiment": experiment_identity, "argv": command,
                                "studio_runtime_sha256": runtime}, sort_keys=True))


def rendering_config(manifest: dict) -> dict:
    config = manifest.get("rendering", {})
    cot, cfg = config.get("cot", "full"), config.get("cfg_scale")
    maximum = config.get("semantic_max_tokens")
    if cot not in {"full", "melody"}:
        raise ValueError("cot must be full or melody")
    if cfg is not None and (not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or not 0 <= cfg <= 20):
        raise ValueError("cfg_scale must be finite and between 0 and 20")
    if maximum is not None and (type(maximum) is not int or maximum < SEMANTIC_DEFAULTS["min_tokens"]):
        raise ValueError("semantic_max_tokens must be an integer >= 200 (native min_tokens)")
    return {"cot": cot, "cfg_scale": cfg, "semantic_max_tokens": maximum}


def prepare(source_plan: Path, lyrics: Path, style: Path, out: Path,
            blueprint_file: Path | None = None, start_section="verse", section_count=2,
            selectors=None, excerpt_lyrics: Path | None = None, seed=1338,
            allow_long=False, arranged_pattern="arpeggio", cot="full", cfg_scale=None,
            semantic_max_tokens=None) -> dict:
    rendering = rendering_config({"rendering": {"cot": cot, "cfg_scale": cfg_scale,
                                                "semantic_max_tokens": semantic_max_tokens}})
    source_abc = source_plan.read_text(encoding="utf-8-sig")
    excerpt, excerpt_info = plan.select_excerpt(source_abc, start_section, section_count, selectors)
    if not allow_long and not 30 <= excerpt_info["duration_s"] <= 60:
        raise ValueError(f"Excerpt is {excerpt_info['duration_s']:.1f}s; choose complete sections totaling 30-60s or explicitly --allow-long")
    lyric_text = (excerpt_lyrics.read_text(encoding="utf-8-sig") if excerpt_lyrics else
                  plan.select_lyrics(lyrics.read_text(encoding="utf-8-sig"), excerpt_info["sections"],
                                     excerpt_info["first_section_occurrence"]))
    style_text = style.read_text(encoding="utf-8-sig")
    bp = read_json(blueprint_file) if blueprint_file else None
    if not style_text.strip() or not lyric_text.strip():
        raise ValueError("Style and excerpt lyrics must be nonempty")
    prepared = {"baseline": excerpt}
    edits = {}
    for variant in VARIANTS[1:]:
        prepared[variant], edits[variant] = plan.edit_instrumental(excerpt, variant, bp,
                                                                 arranged_pattern if variant == "arranged" else "arpeggio")
    conversions = {}
    if cot == "melody":
        excerpt_info["vocal_identity_before_cot_conversion"] = excerpt_info["vocal_identity"]
        for variant in prepared:
            prepared[variant], conversions[variant] = plan.strip_music_chords(prepared[variant])
            if variant in edits:
                edits[variant]["vocal_header_unchanged"] = False
                edits[variant]["chords_in_vocal_unchanged"] = False
                edits[variant]["note_content_unchanged_by_cot_conversion"] = True
                edits[variant]["cot_conversion"] = conversions[variant]
                edits[variant]["vocal_header_sha256"] = plan.vocal_identity(*plan.parse(prepared[variant]))
        excerpt_info["vocal_identity"] = plan.vocal_identity(*plan.parse(prepared["baseline"]))
    # Build every variant successfully before touching an existing experiment.
    identity_data = {"source_plan": digest(source_plan), "excerpt": excerpt,
                                   "variants": prepared, "lyrics": lyric_text, "style": style_text,
                                   "blueprint": bp, "seed": seed}
    # Keep legacy default identities stable; non-default generation settings
    # must be explicit members of the prepared-input identity.
    identity_rendering = {"cot": cot, "cfg_scale": cfg_scale}
    if semantic_max_tokens is not None:
        identity_rendering["semantic_max_tokens"] = semantic_max_tokens
    if identity_rendering != {"cot": "full", "cfg_scale": None}:
        identity_data["rendering"] = identity_rendering
    identity = plan.sha(json.dumps(identity_data, sort_keys=True, ensure_ascii=False))
    manifest_file = out / "manifest.json"
    if manifest_file.exists():
        existing = read_json(manifest_file)
        if existing.get("experiment_identity") == identity:
            return existing
        raise ValueError("Experiment already exists with different inputs; choose a new --out-dir")
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_plan.abc").write_text(source_abc, encoding="utf-8")
    (out / "lyrics.txt").write_text(lyric_text, encoding="utf-8")
    (out / "style.txt").write_text(style_text, encoding="utf-8")
    if bp:
        write_json(out / "blueprint.json", bp)
    variants = {}
    for variant, abc in prepared.items():
        p = out / "plans" / f"{variant}.abc"
        p.parent.mkdir(exist_ok=True)
        p.write_text(abc, encoding="utf-8")
        variants[variant] = {"plan": f"plans/{variant}.abc", "plan_sha256": digest(p),
                             "seed": seed, "edit": edits.get(variant, {"source_excerpt_unchanged": not bool(conversions),
                                                                         "cot_conversion": conversions.get(variant)}),
                             "audio": f"renders/{variant}/audio.wav"}
    mapping = edits["arranged"]["mapping"]
    manifest = {
        "schema": 1, "experiment_identity": identity,
        "workflow": "generic_ins_control" if mapping["provenance"] == "generic_control" else "reference_mapped_arrangement_control",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_plan": str(source_plan.resolve()), "source_plan_sha256": digest(source_plan),
        "source_lyrics": str(lyrics.resolve()), "source_style": str(style.resolve()),
        "excerpt": excerpt_info, "lyrics_selection": "explicit_file" if excerpt_lyrics else "matching_original_blocks_including_prechorus",
        "lyrics_sha256": digest(out / "lyrics.txt"), "style_sha256": digest(out / "style.txt"),
        "blueprint_status": (bp or {}).get("annotation_status", "no_reference_blueprint"),
        "arrangement_mapping": mapping,
        "arranged_pattern": arranged_pattern,
        "rendering": rendering,
        "cot_conversion": {"status": "applied" if conversions else "not_needed_cot_full", "variants": conversions},
        "variants": variants,
        "comparison_design": "Same Vocal/header/chords/lyrics/style/seed; only Ins note content changes. Sparse/full test control; arranged tests verse-to-chorus contrast.",
        "claim_status": "unvalidated_experiment_no_quality_claim",
        "promotion_requires": ["Reference section timings and roles reviewed by a listener",
                               "Ins changes audible in multiple same-seed comparisons",
                               "Lyrics, vocal naturalness and mix quality accepted by a listener",
                               "Blind listening improves over baseline without regressions"],
    }
    ratings = {"schema": 1, "scale": "1 (poor) to 10 (excellent); null means not listened",
               "status": "not_listened", "listener": None, "reference_annotations_reviewed": False,
               "dimensions": ["verse_chorus_contrast", "groove", "instrument_entry_clarity", "vocal_naturalness", "lyric_intelligibility", "hook_memorability", "mix_cleanliness", "overall"],
               "scores": {v: {d: None for d in ["verse_chorus_contrast", "groove", "instrument_entry_clarity", "vocal_naturalness", "lyric_intelligibility", "hook_memorability", "mix_cleanliness", "overall"]} for v in VARIANTS},
               "preferred_variant": None, "ins_control_audible": None, "notes": "", "promotion_decision": "pending_human_listening"}
    write_json(out / "ratings.json", ratings)
    write_json(manifest_file, manifest)
    return manifest


def checked_manifest(experiment: Path) -> dict:
    manifest = read_json(experiment / "manifest.json")
    rendering_config(manifest)
    for path, expected in [(experiment / "lyrics.txt", manifest["lyrics_sha256"]),
                           (experiment / "style.txt", manifest["style_sha256"])]:
        if digest(path) != expected:
            raise ValueError(f"Prepared input changed: {path}; create a new experiment")
    for variant in manifest["variants"].values():
        path = experiment / variant["plan"]
        if digest(path) != variant["plan_sha256"]:
            raise ValueError(f"Prepared plan changed: {path}; create a new experiment")
    return manifest


def render_commands(experiment: Path, variants: list[str] | None = None,
                    python: Path = YUE_PYTHON, budget=12.0, query_chunk=1024,
                    backend="torch-eager", offload=True) -> list[tuple[str, list[str]]]:
    manifest = checked_manifest(experiment)
    chosen = variants or list(manifest["variants"])
    if len(set(chosen)) != len(chosen) or any(v not in manifest["variants"] for v in chosen):
        raise ValueError("Unknown or duplicate variant")
    if budget <= 0 or query_chunk < 1:
        raise ValueError("Budget/query chunk must be positive")
    commands = []
    for variant in chosen:
        command = [str(python.resolve()), "-I", "-u", str(Path(__file__).with_name("render_worker.py")),
                   "--experiment", str(experiment.resolve()), "--variant", variant,
                   "--budget", str(budget), "--nar-query-chunk", str(query_chunk), "--backend", backend]
        if offload:
            command.append("--offload-ar")
        commands.append((variant, command))
    return commands


def dispatch(experiment: Path, variants=None, python=YUE_PYTHON, budget=12.0,
             query_chunk=1024, backend="torch-eager", offload=True, dry_run=False,
             force=False, timeout_minutes=40) -> dict:
    commands = render_commands(experiment, variants, python, budget, query_chunk, backend, offload)
    if dry_run:
        return {"status": "dry_run_no_gpu_started", "commands": [dict(variant=v, argv=c) for v, c in commands]}
    if not python.exists():
        raise FileNotFoundError(f"YuE2 Python missing: {python}")
    manifest = checked_manifest(experiment)
    outcomes = {}
    for variant, command in commands:
        directory = experiment / "renders" / variant
        directory.mkdir(parents=True, exist_ok=True)
        state_file = directory / "state.json"
        request_identity = request_fingerprint(manifest["experiment_identity"], command)
        state = read_json(state_file) if state_file.exists() else {}
        wav = directory / "audio.wav"
        result_file = directory / "result.json"
        if (not force and state.get("status") == "complete" and
                state.get("request_identity") == request_identity and wav.exists() and result_file.exists() and
                digest(wav) == state.get("audio_sha256")):
            outcomes[variant] = {"status": "skipped_verified_complete", "audio": str(wav.resolve())}
            continue
        write_json(state_file, {"status": "running", "request_identity": request_identity,
                                "command": command, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        log = directory / "render.log"
        started = time.monotonic()
        print(f"{variant}: rendering in isolated process -> {log}", flush=True)
        try:
            with log.open("w", encoding="utf-8") as output:
                child = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                       timeout=timeout_minutes * 60)
            if child.returncode or not wav.exists() or not result_file.exists():
                raise RuntimeError(f"Renderer exit {child.returncode}; inspect {log}")
            result = read_json(result_file)
            if result.get("experiment_identity") != manifest["experiment_identity"]:
                raise RuntimeError("Renderer result belongs to a different experiment")
            truncated = result.get("truncated", {})
            if any(bool(x) for x in truncated.values()):
                raise RuntimeError(f"Renderer output truncated: {truncated}")
            current = {"status": "complete", "request_identity": request_identity,
                       "audio_sha256": digest(wav), "elapsed_s": round(time.monotonic() - started, 2),
                       "audio": str(wav.resolve()), "command": command}
        except Exception as exc:
            current = {"status": "failed", "request_identity": request_identity,
                       "elapsed_s": round(time.monotonic() - started, 2), "error": str(exc), "log": str(log.resolve())}
        write_json(state_file, current)
        outcomes[variant] = current
    return {"status": "failed" if any(v["status"] == "failed" for v in outcomes.values()) else "complete",
            "variants": outcomes}


def duration_validation(actual: float, planned: float, tolerance=.10) -> dict:
    if planned <= 0:
        raise ValueError("Planned duration must be positive")
    relative = (actual - planned) / planned
    needs_review = abs(relative) > tolerance
    return {"planned_duration_s": planned, "actual_duration_s": actual,
            "delta_s": round(actual - planned, 4), "relative_error": round(relative, 6),
            "relative_error_percent": round(relative * 100, 3), "tolerance_relative": tolerance,
            "technical_review_required": needs_review,
            "status": "technical_review_required" if needs_review else "within_duration_tolerance_alignment_still_unverified",
            "symbolic_section_metrics_allowed": not needs_review}


def report(experiment: Path, target_dbfs=-20.0) -> dict:
    from .audio import decode, descriptors, loudness_match, native_file_levels, probe

    manifest = checked_manifest(experiment)
    available, states, excluded, executions, reuse_lineage = {}, {}, {}, {}, {}
    for name, variant in manifest["variants"].items():
        path = experiment / variant["audio"]
        state_file = experiment / "renders" / name / "state.json"
        state = read_json(state_file) if state_file.exists() else {"status": "not_rendered"}
        states[name] = state.get("status", "unknown")
        executions[name] = state.get("execution", "inference" if state.get("status") == "complete" else "pending")
        if state.get("reuse_lineage"):
            reuse_lineage[name] = state["reuse_lineage"]
        if state.get("status") == "complete" and path.exists() and digest(path) == state.get("audio_sha256"):
            available[name] = path
        elif path.exists():
            excluded[name] = "Audio present without a verified complete state; excluded from listening pack"
    metrics, levels = {}, None
    if available:
        levels = loudness_match(available, experiment / "listening", target_dbfs)
        baseline_abc = (experiment / manifest["variants"]["baseline"]["plan"]).read_text(encoding="utf-8")
        header, sections = plan.parse(baseline_abc)
        clock = plan.timing(header)
        boundaries, t = [], 0.0
        for sec in sections:
            duration = sum(len(plan.bars(c.vocal, clock["units_per_bar"])) for c in sec.chunks) * clock["seconds_per_bar"]
            boundaries.append((plan.section_name(sec.marker), t, t + duration))
            t += duration
        for name, path in available.items():
            info = probe(path)
            timing_check = duration_validation(info["duration_s"], manifest["excerpt"]["duration_s"])
            y = decode(path, 22050)
            whole = descriptors(y, 22050)
            whole["resampled_mono_level_diagnostics"] = {
                key: whole[key] for key in ("rms_dbfs", "sample_peak_dbfs", "crest_db")}
            whole.update(native_file_levels(path))
            section_metrics = []
            for label, start, end in boundaries:
                if timing_check["technical_review_required"]:
                    section_metrics.append({"section": label, "symbolic_start_s": round(start, 3),
                                            "symbolic_end_s": round(end, 3), "values": None,
                                            "status": "disabled_duration_drift_exceeds_tolerance_requires_audio_alignment"})
                    continue
                if start >= len(y) / 22050:
                    section_metrics.append({"section": label, "missing": True})
                    continue
                values = descriptors(y[int(start * 22050):min(int(end * 22050), len(y))], 22050)
                section_metrics.append({"section": label, "symbolic_start_s": round(start, 3),
                                        "symbolic_end_s": round(end, 3), "values": values})
            metrics[name] = {"audio": str(path.resolve()), "probe": info,
                             "symbolic_duration_delta_s": round(info["duration_s"] - manifest["excerpt"]["duration_s"], 3),
                             "duration_validation": timing_check,
                             "section_timing_status": ("disabled_duration_drift_requires_audio_alignment" if timing_check["technical_review_required"] else "symbolic_approximation_requires_listening_alignment"),
                             "whole_mix": whole, "sections": section_metrics}
    comparison = {}
    if "baseline" in metrics:
        baseline = metrics["baseline"]["whole_mix"]
        for name, row in metrics.items():
            if name == "baseline":
                continue
            values = row["whole_mix"]
            comparison[name] = {"rms_delta_db": round(values["rms_dbfs"] - baseline["rms_dbfs"], 3),
                                "mix_transient_rate_delta_per_s": round(values["mix_transient_events_per_s"] - baseline["mix_transient_events_per_s"], 4),
                                "interpretation": "Descriptive mix difference; not evidence that Ins caused an improvement."}
    ratings = read_json(experiment / "ratings.json")
    technical_reviews = {name: row["duration_validation"] for name, row in metrics.items()
                         if row["duration_validation"]["technical_review_required"]}
    result = {"schema": 1, "experiment_identity": manifest["experiment_identity"],
              "status": "technical_review_required" if technical_reviews else "awaiting_human_listening" if available else "awaiting_render",
              "render_states": states, "excluded_audio": excluded, "loudness_match": levels,
              "render_executions": executions, "reuse_lineage": reuse_lineage,
              "rendering": rendering_config(manifest),
              "metrics": metrics, "comparison_to_baseline": comparison, "human_ratings": ratings,
              "quality_claim": "No high-score claim: automated descriptors are not a validated perceptual score.",
              "reference_annotation_status": manifest["blueprint_status"],
              "arrangement_mapping": manifest.get("arrangement_mapping", {"provenance": "legacy_unreported_requires_review"}),
              "technical_review_variants": technical_reviews,
              "promotion_status": "technical_review_required" if technical_reviews else "manual_decision_required"}
    write_json(experiment / "report.json", result)
    lines = ["# Reference arrangement listening experiment", "",
             f"State: {result['status']}. Reference annotations: {manifest['blueprint_status']}.", "",
             f"Arrangement provenance: {result['arrangement_mapping']['provenance']}.", "",
             f"Rendering: cot={result['rendering']['cot']}, cfg_scale={result['rendering']['cfg_scale']}, semantic_max_tokens={result['rendering']['semantic_max_tokens']} (None means native default).", "",
             "No proxy score certifies quality, hook identity, instrument identity, or a successful remake.", "",
             "| Variant | Render | Listening file |", "| --- | --- | --- |"]
    for name in manifest["variants"]:
        file = f"[listen](listening/{name}.wav)" if name in available else "pending"
        description = f"{states[name]} (input-equivalent reuse)" if executions[name] == "input_equivalent_reuse" else states[name]
        lines.append(f"| {name} | {description} | {file} |")
    if reuse_lineage:
        lines += ["", "Input-equivalent reuse: these files were copied from verified original inference with identical plan/lyrics/style bytes, seed and runtime parameters. No new inference was performed; this is not an independent replication."]
    lines += ["", "| Variant | Plan seconds | Actual seconds | Relative duration error | Timing QA |",
              "| --- | --- | --- | --- | --- |"]
    for name, row in metrics.items():
        check = row["duration_validation"]
        lines.append(f"| {name} | {check['planned_duration_s']:.3f} | {check['actual_duration_s']:.3f} | {check['relative_error_percent']:+.3f}% | {check['status']} |")
    if technical_reviews:
        lines += ["", "Technical review required: actual duration differs from the plan by more than 10%. Symbolic section metrics for those variants are disabled; align the actual audio before judging verse/chorus differences. No time stretching or trimming has concealed this drift."]
    lines += ["", "The listening files share one safe mono RMS target via linear gain. This is not LUFS matching.",
              "Section measurements use symbolic boundaries and require audible alignment checks.", "",
              "Edit ratings.json after listening: contrast, groove, instrument entry, vocal naturalness, diction, hook, mix and overall.",
              "Check sparse/full first: can the renderer audibly follow Ins? Compare arranged to baseline next.",
              "If Ins cannot reliably control the accompaniment, use a MIDI/DAW stem arrangement rather than extending this experiment to full songs.", ""]
    (experiment / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result
