"""Explicit reuse of verified inference when actual model inputs are identical.

Changing the explanatory reference mapping alone does not create another model
request. Reuse is recorded as reuse, with the original inference evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

from .workflow import (ROOT, YUE_PYTHON, SEMANTIC_DEFAULTS, checked_manifest, digest, read_json,
                       render_commands, rendering_config, request_fingerprint, write_json)


def _native_identity(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _runtime_model_identity(python: Path) -> dict:
    """Read code and hash model files, without importing Torch or using the GPU."""
    package = python.resolve().parent.parent / "Lib" / "site-packages" / "yue2"
    if not (package / "pipeline.py").exists():
        raise ValueError("Cannot verify the installed YuE2 runtime for this Python")
    runtime = _native_identity({p.name: digest(p) for p in sorted(package.glob("*.py"))})
    weights = {}
    for role, name in (("mot", "YuE2-3B"), ("vae", "YuE2-Vae")):
        directory = ROOT / "models" / "yue2" / name
        files = sorted(directory.glob("*.safetensors"))
        if not files:
            raise ValueError(f"Cannot verify model files in {directory}")
        weights[role] = {"files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in files},
                         "config_sha256": digest(directory / "config.json")}
    return {"runtime_sha256": runtime, "weights": weights}


def _runtime_arguments(argv: list[str]) -> dict:
    if len(argv) < 4 or argv[1:3] != ["-I", "-u"]:
        raise ValueError("Source command is not a recognized isolated render worker")
    expected_worker = Path(__file__).with_name("render_worker.py").resolve()
    if Path(argv[3]).resolve() != expected_worker:
        raise ValueError("Source command used a different worker")
    values = {"python": str(Path(argv[0]).resolve()).casefold(),
              "worker": str(expected_worker).casefold(), "offload_ar": False}
    index = 4
    converters = {"--experiment": str, "--variant": str, "--budget": float,
                  "--nar-query-chunk": int, "--backend": str}
    seen = set()
    while index < len(argv):
        key = argv[index]
        if key in seen:
            raise ValueError("Duplicate source command option")
        seen.add(key)
        if key == "--offload-ar":
            values["offload_ar"] = True
            index += 1
            continue
        if key not in converters or index + 1 >= len(argv):
            raise ValueError(f"Unrecognized source runtime option: {key}")
        values[key.lstrip("-").replace("-", "_")] = converters[key](argv[index + 1])
        index += 2
    if not set(converters).issubset(seen):
        raise ValueError("Incomplete source runtime parameters")
    return values


def _no_truncation(result: dict, label: str):
    truncated = result.get("truncated")
    if not isinstance(truncated, dict) or any(truncated.get(key) is not False for key in ("abc", "semantic")):
        raise ValueError(f"{label} lacks an explicit untruncated ABC and semantic result")


def reuse_renders(source: Path, experiment: Path, variants: list[str],
                  python: Path = YUE_PYTHON, budget=12.0, query_chunk=1024,
                  backend="torch-eager", offload=True) -> dict:
    source, experiment = source.resolve(), experiment.resolve()
    if source == experiment:
        raise ValueError("Source and destination experiment must differ")
    if not variants:
        raise ValueError("Select at least one variant explicitly")
    source_manifest = checked_manifest(source)
    target_manifest = checked_manifest(experiment)
    source_rendering, target_rendering = rendering_config(source_manifest), rendering_config(target_manifest)
    if source_rendering != target_rendering:
        raise ValueError("cot/cfg_scale/semantic_max_tokens generation settings differ; reuse refused")
    commands = render_commands(experiment, variants, python, budget, query_chunk, backend, offload)
    ready = []
    verified_runtime = None
    for variant, command in commands:
        if variant not in source_manifest["variants"]:
            raise ValueError(f"Source has no variant {variant}")
        original = source_manifest["variants"][variant]
        target = target_manifest["variants"][variant]
        input_hashes = {}
        for label, a, b in (("plan", source/original["plan"], experiment/target["plan"]),
                            ("lyrics", source/"lyrics.txt", experiment/"lyrics.txt"),
                            ("style", source/"style.txt", experiment/"style.txt")):
            first, second = digest(a), digest(b)
            if first != second:
                raise ValueError(f"{variant}: {label} input bytes differ; reuse refused")
            input_hashes[label] = first
        if original["seed"] != target["seed"]:
            raise ValueError(f"{variant}: seed differs; reuse refused")
        directory = source/"renders"/variant
        state_file, result_file = directory/"state.json", directory/"result.json"
        if not state_file.exists() or not result_file.exists():
            raise ValueError(f"{variant}: source has no completed render evidence")
        state, result = read_json(state_file), read_json(result_file)
        if state.get("status") != "complete":
            raise ValueError(f"{variant}: source render is not complete")
        if state.get("execution") == "input_equivalent_reuse":
            raise ValueError("Use the original inference experiment as source; reuse chains are not accepted")
        audio = directory/"audio.wav"
        if not audio.exists() or digest(audio) != state.get("audio_sha256"):
            raise ValueError(f"{variant}: source audio hash does not match completed state")
        if (result.get("experiment_identity") != source_manifest["experiment_identity"] or
                result.get("variant") != variant or result.get("seed") != original["seed"]):
            raise ValueError(f"{variant}: source result identity/seed does not match")
        if rendering_config(result) != source_rendering:
            raise ValueError(f"{variant}: source result cot/cfg_scale/semantic_max_tokens differs from its manifest")
        _no_truncation(result, f"{variant} source wrapper")
        actual = _runtime_arguments(state.get("command", []))
        planned = _runtime_arguments(command)
        if Path(actual.pop("experiment")).resolve() != source or actual.pop("variant") != variant:
            raise ValueError(f"{variant}: source command belongs to another experiment")
        planned.pop("experiment")
        planned.pop("variant")
        if actual != planned:
            raise ValueError(f"{variant}: runtime parameters differ; reuse refused")
        artifacts = directory/"artifacts"
        native_file = artifacts/"result.json"
        if not native_file.exists():
            raise ValueError(f"{variant}: native YuE2 artifacts are missing")
        native = read_json(native_file)
        if native.get("status") != "complete":
            raise ValueError(f"{variant}: native artifact is not complete")
        _no_truncation(native, f"{variant} native artifact")
        for name in ("request.json", "config.json", "score.abc"):
            p = artifacts/name
            if not p.exists() or digest(p) != native.get("artifacts", {}).get(name, {}).get("sha256"):
                raise ValueError(f"{variant}: native {name} hash is missing or invalid")
        request = read_json(artifacts/"request.json")
        for key, filename in (("style", "style.txt"), ("lyrics", "lyrics.txt"), ("abc", original["plan"])):
            if request.get(key) != (source/filename).read_text(encoding="utf-8"):
                raise ValueError(f"{variant}: actual native {key} differs from prepared input")
        if (request.get("seed") != original["seed"] or request.get("cot") != source_rendering["cot"] or
                request.get("cfg_scale") != source_rendering["cfg_scale"]):
            raise ValueError(f"{variant}: native sampling request differs from this worker's defaults")
        if (artifacts/"score.abc").read_text(encoding="utf-8") != request["abc"]:
            raise ValueError(f"{variant}: native score differs from the request")
        config = read_json(artifacts/"config.json")
        effective_cfg = source_rendering["cfg_scale"] if source_rendering["cfg_scale"] is not None else 1.0
        expected_semantic = dict(SEMANTIC_DEFAULTS)
        if source_rendering["semantic_max_tokens"] is not None:
            expected_semantic["max_tokens"] = source_rendering["semantic_max_tokens"]
        expected_overrides = {}
        if effective_cfg != 1.0:
            expected_overrides["cfg_scale"] = effective_cfg
        if expected_semantic != SEMANTIC_DEFAULTS:
            expected_overrides["semantic"] = expected_semantic
        if (config.get("backend") != planned["backend"] or
                config.get("memory_budget_gib") != planned["budget"] or
                config.get("offload_ar") != planned["offload_ar"] or
                config.get("cot") != source_rendering["cot"] or config.get("cfg_scale") != effective_cfg or
                config.get("quantization") != "none" or config.get("overrides") != expected_overrides or
                config.get("generation", {}).get("semantic") != expected_semantic):
            raise ValueError(f"{variant}: native runtime configuration differs from the planned worker")
        if verified_runtime is None:
            verified_runtime = _runtime_model_identity(python)
        if config.get("runtime_sha256") != verified_runtime["runtime_sha256"] or native.get("weights") != verified_runtime["weights"]:
            raise ValueError(f"{variant}: installed YuE2 runtime or model identity changed")
        destination = experiment/"renders"/variant
        destination_state = destination/"state.json"
        previous = read_json(destination_state) if destination_state.exists() else {}
        if destination.exists() and any(destination.iterdir()):
            if (previous.get("execution") == "input_equivalent_reuse" and previous.get("status") == "complete" and
                    previous.get("reuse_lineage", {}).get("source_experiment_identity") == source_manifest["experiment_identity"] and
                    previous.get("audio_sha256") == state["audio_sha256"] and (destination/"audio.wav").exists() and
                    digest(destination/"audio.wav") == state["audio_sha256"]):
                ready.append((variant, "already_reused", None))
                continue
            raise ValueError(f"{variant}: destination already has render files; refusing to replace them")
        lineage = {"source_experiment": str(source), "source_experiment_identity": source_manifest["experiment_identity"],
                   "source_variant": variant, "source_audio": str(audio), "source_audio_sha256": state["audio_sha256"],
                   "source_state_sha256": digest(state_file), "source_result_sha256": digest(result_file),
                   "source_native_result_sha256": digest(native_file), "source_native_identity": native.get("identity"),
                   "source_inference_command": state["command"], "actual_runtime_parameters": planned,
                   "verified_input_sha256": input_hashes, "seed": original["seed"],
                   "rendering": source_rendering,
                   "verified_yue2_runtime_and_models": verified_runtime,
                   "verification": "input_file_bytes_seed_runtime_parameters_native_request_models_and_untruncated_audio_verified",
                   "inference_performed_for_destination": False,
                   "reused_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        ready.append((variant, "ready", (audio, directory, command, lineage, result)))
    # No destination writes until every requested source arm passed preflight.
    outcomes = {}
    for variant, status, data in ready:
        if status == "already_reused":
            outcomes[variant] = {"status": "skipped_verified_input_equivalent_reuse"}
            continue
        audio, directory, command, lineage, original_result = data
        destination = experiment/"renders"/variant
        destination.mkdir(parents=True, exist_ok=True)
        copy_started = time.monotonic()
        shutil.copy2(audio, destination/"audio.wav")
        if digest(destination/"audio.wav") != lineage["source_audio_sha256"]:
            raise ValueError(f"{variant}: copied audio failed hash verification; no complete state was written")
        evidence = destination/"reuse_source"
        evidence.mkdir(exist_ok=True)
        for name in ("state.json", "result.json"):
            shutil.copy2(directory/name, evidence/name)
        native_evidence = evidence/"artifacts"
        native_evidence.mkdir(exist_ok=True)
        for name in ("result.json", "request.json", "config.json", "score.abc"):
            shutil.copy2(directory/"artifacts"/name, native_evidence/name)
        result = {"experiment_identity": target_manifest["experiment_identity"], "variant": variant,
                  "seed": lineage["seed"], "truncated": original_result["truncated"],
                  "rendering": source_rendering,
                  "execution": "input_equivalent_reuse", "inference_performed": False, "reuse_lineage": lineage}
        write_json(destination/"result.json", result)
        state = {"status": "complete", "execution": "input_equivalent_reuse", "inference_performed": False,
                 "request_identity": request_fingerprint(target_manifest["experiment_identity"], command),
                 "audio_sha256": lineage["source_audio_sha256"], "audio": str(destination/"audio.wav"),
                 "planned_equivalent_command_not_executed": command, "reuse_lineage": lineage,
                 "copy_elapsed_s": round(time.monotonic() - copy_started, 3)}
        write_json(destination/"state.json", state)
        outcomes[variant] = {"status": "input_equivalent_reuse", "audio": str(destination/"audio.wav"),
                             "source_experiment": str(source), "inference_performed": False}
    return {"status": "complete", "execution": "explicit_verified_reuse_no_new_inference", "variants": outcomes}
