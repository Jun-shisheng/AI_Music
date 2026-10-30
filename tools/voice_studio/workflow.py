from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from . import audio, band_restore, highnote_qa, registers
from .assets import ASSETS
from .download_models import verify as verify_asset

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT / "_archive/outputs_history/2026-10-16_epic_hook/selected/04_iron_rain_raw.wav"
VERSION = 2
OFFICIAL_SIZES={"svc":2793965154,"svs":2818092278}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value) or value in {".", ".."}:
        raise ValueError("ID must contain 1–80 letters, numbers, dots, hyphens or underscores")
    return value


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def backend_identity(python:Path)->dict:
    """Query package metadata in the actual interpreter without importing torch/CUDA."""
    packages=["torch","torchaudio","transformers","tokenizers","numpy"]
    result=dict(requested_interpreter=str(python.resolve()),packages={name:None for name in packages})
    if not python.is_file():
        return {**result,"status":"interpreter_missing"}
    script="""import importlib.metadata as metadata, json, sys
names=['torch','torchaudio','transformers','tokenizers','numpy']
versions={}
for name in names:
    try:
        versions[name]=metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name]=None
print(json.dumps(dict(interpreter=sys.executable,python_version=sys.version.split()[0],packages=versions)))
"""
    try:
        completed=subprocess.run([str(python),"-c",script],check=True,capture_output=True,text=True,timeout=30)
        result.update(json.loads(completed.stdout),status="metadata_read")
    except (subprocess.SubprocessError,ValueError,OSError) as exc:
        result.update(status="metadata_failed",error=str(exc))
    return result


def whisper_identity(models:Path)->dict:
    directory=models/"whisper-base"
    return {name:dict(path=str((directory/name).resolve()),sha256=audio.digest(directory/name)
        if (directory/name).is_file() else None) for name in ASSETS["openai/whisper-base"]["files"]}


def run(command: list[str], log: Path, *, cwd: Path | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(command, ensure_ascii=False) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=cwd or PROJECT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-2400:]
        raise RuntimeError(f"Command exited {result.returncode}; log: {log}\n{tail}")


def worker(args, task: str, spec: dict, directory: Path, *, audio_backend: bool = False) -> None:
    spec_path = directory / f"{task}_spec.json"
    audio.atomic_json(spec_path, spec)
    python = args.audio_python if audio_backend else args.backend_python
    if not python.is_file():
        raise FileNotFoundError(f"Required interpreter does not exist: {python}")
    run([str(python), str(Path(__file__).with_name("worker.py")), task, "--spec", str(spec_path)],
        directory / f"{task}.log")


def f0(args, files: list[tuple[Path, Path]], directory: Path, *,
       method: str = "dio", f0_floor: float = 55, f0_ceil: float = 1600) -> None:
    descriptor = dict(method=method, f0_floor=f0_floor, f0_ceil=f0_ceil, version=VERSION)
    pending=[]
    identities={}
    for source,output in files:
        marker=output.with_suffix(output.suffix+".cache.json")
        identity=audio.fingerprint(dict(audio_sha256=audio.digest(source), **descriptor))
        identities[str(output)]=(identity,marker)
        valid=False
        if output.exists() and marker.exists():
            cached=load(marker)
            valid=cached.get("key")==identity and cached.get("sha256")==audio.digest(output)
        if not valid:
            pending.append(dict(audio=str(source),output=str(output)))
    if pending:
        worker(args, "f0", dict(files=pending, **descriptor), directory, audio_backend=True)
        for item in pending:
            output=Path(item["output"])
            identity,marker=identities[item["output"]]
            audio.atomic_json(marker,dict(key=identity,sha256=audio.digest(output)))


def slice_contour(contour: np.ndarray, start_sample: int, n_samples: int, rate: int,
                  model_rate: int = 50) -> np.ndarray:
    """Slice a whole-file 50 Hz contour for one chunk, on the model's frame grid."""
    start = int(round(start_sample / rate * model_rate))
    frames = int(np.ceil(n_samples / rate * model_rate))
    value = np.asarray(contour)[start:start + frames]
    return np.pad(value, (0, max(0, frames - len(value))))[:frames].astype("float32")


def separate(args, source: Path, directory: Path) -> tuple[Path, Path]:
    """Cache by complete audio identity + separator settings, never by basename."""
    key = audio.fingerprint(dict(sha256=audio.digest(source), model="htdemucs", shifts=0, version=VERSION))
    cache = args.workspace / "cache/separation" / key
    stems = cache / "htdemucs" / source.stem
    vocals, instrumental = stems / "vocals.wav", stems / "no_vocals.wav"
    marker = cache / "complete.json"
    if vocals.exists() and instrumental.exists() and marker.exists():
        state = load(marker)
        if all(audio.digest(p) == state.get(p.name) for p in (vocals, instrumental)):
            return vocals, instrumental
    if not args.audio_python.exists():
        raise FileNotFoundError(f"Demucs interpreter missing: {args.audio_python}")
    run([str(args.audio_python), "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
         "--shifts", "0", "-d", args.separation_device, "-o", str(cache), str(source)], directory / "separation.log")
    if not vocals.exists() or not instrumental.exists():
        raise RuntimeError("Demucs returned without both vocals.wav and no_vocals.wav")
    audio.atomic_json(marker, {p.name: audio.digest(p) for p in (vocals, instrumental)})
    return vocals, instrumental


def model_spec(args, kind: str) -> dict:
    root = args.soulx_root.resolve()
    model = args.models / "SoulX-Singer" / ("model-svc.pt" if kind == "svc" else "model.pt")
    config = root / "soulxsinger/config/soulxsinger.yaml"
    whisper = args.models / "whisper-base"
    required = [args.backend_python, root / f"cli/{'inference_svc' if kind == 'svc' else 'inference'}.py", model, config]
    if kind == "svc":
        required += [whisper/name for name in ASSETS["openai/whisper-base"]["files"]]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Backend is not prepared. Missing: " + "; ".join(missing))
    if model.stat().st_size != OFFICIAL_SIZES[kind]:
        raise ValueError(f"Official {kind} checkpoint is incomplete or unexpected: {model.stat().st_size} bytes, expected {OFFICIAL_SIZES[kind]}")
    model_hash=audio.digest(model)
    if model_hash!=ASSETS["Soul-AILab/SoulX-Singer"]["files"][model.name]["sha256"]:
        raise ValueError(f"Official checkpoint SHA-256 mismatch: {model}; verify the download")
    if kind=="svc":
        for name,expected in ASSETS["openai/whisper-base"]["files"].items():
            verify_asset(whisper/name,expected)
    return dict(models=str(args.models.resolve()), soulx_root=str(root), model=str(model.resolve()),
        config=str(config), whisper=str(whisper.resolve()), device=args.device,
        fp16=not args.fp32 and args.device.startswith("cuda"), steps=args.steps, cfg=args.cfg,
        seed=args.seed, pitch_shift=args.pitch_shift,
        model_sha256=model_hash, config_sha256=audio.digest(config))


def profile(args) -> dict:
    directory = args.workspace / "profiles" / safe_id(args.voice_id)
    path = directory / "profile.json"
    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    identity = audio.digest(source)
    if path.exists():
        previous = load(path)
        if previous["source_sha256"] != identity:
            raise ValueError("Voice ID already belongs to another source; choose a new versioned voice_id")
        if previous.get("preparation_status") == "reference_ready" or previous.get("status") == "reference_ready":
            if args.refresh:
                raise ValueError("Prepared singer reference is immutable; choose a new versioned voice_id to reselect it")
            for item in previous["artifacts"].values():
                if not Path(item["path"]).exists() or audio.digest(Path(item["path"])) != item["sha256"]:
                    raise ValueError("Profile artifact changed; create a new version instead of silently changing singer identity")
            return previous
    directory.mkdir(parents=True, exist_ok=True)
    original = directory / ("original" + source.suffix.lower())
    if not original.exists():
        shutil.copy2(source, original)
    if audio.digest(original)!=identity:
        raise ValueError("Archived source differs from the selected source; use a new voice ID")
    result = dict(schema_version=VERSION, voice_id=args.voice_id, status="registered", created_at=now(),
        source=str(source), archived_original=str(original), source_sha256=identity,
        note="Reference is derived from the selected AI-generated original. Identity consistency requires cross-song listening.")
    audio.atomic_json(path, result)
    if not args.execute:
        result["next_action"] = "Run profile with --execute to separate vocals and prepare the actual reference"
        audio.atomic_json(path, result)
        return result
    try:
        result["status"] = "extracting_reference"
        audio.atomic_json(path, result)
        if args.vocals:
            vocal_source = args.vocals.resolve()
        else:
            vocal_source, _ = separate(args, source, directory)
        source_info,vocal_info=audio.info(source),audio.info(vocal_source)
        if abs(source_info["duration_seconds"]-vocal_info["duration_seconds"])>.08:
            raise ValueError("Source and supplied vocal stem durations differ; supply the matching cropped source and preserve parent provenance with --source-manifest")
        result.update(source_audio_info=source_info,vocal_stem_audio_info=vocal_info,
            vocal_stem_source=str(vocal_source),vocal_stem_sha256=audio.digest(vocal_source))
        if args.source_manifest:
            provenance=directory/"source_provenance.json"
            shutil.copy2(args.source_manifest,provenance)
            result["source_provenance"]=dict(path=str(provenance.resolve()),sha256=audio.digest(provenance))
        if args.parent_source:
            parent=args.parent_source.resolve()
            archived_parent=directory/("original_parent"+parent.suffix.lower())
            if not archived_parent.exists():
                shutil.copy2(parent,archived_parent)
            parent_hash=audio.digest(parent)
            if audio.digest(archived_parent)!=parent_hash:
                raise ValueError("Archived parent original no longer matches --parent-source")
            result["parent_original"]=dict(source=str(parent),path=str(archived_parent.resolve()),
                sha256=parent_hash,audio_info=audio.info(parent))
        dry = audio.canonical(vocal_source, directory / "source_vocals.wav")
        data, rate = audio.read(dry)
        candidates = audio.reference_candidates(data, rate, args.reference_seconds)
        for index, candidate in enumerate(candidates):
            candidate_path=directory/f"reference_candidate_{index+1:02d}.wav"
            audio.write(candidate_path, data[candidate["start"]:candidate["end"]],rate)
            candidate.update(path=str(candidate_path.resolve()),start_seconds=round(candidate["start"]/rate,3),
                duration_seconds=round((candidate["end"]-candidate["start"])/rate,3))
        if args.reference_start is None:
            a, b = candidates[0]["start"],candidates[0]["end"]
        else:
            a = round(args.reference_start * rate)
            b = a + round(args.reference_seconds * rate)
            if a < 0 or b > len(data):
                raise ValueError("Reference excerpt lies outside the source")
        reference = directory / "reference.wav"
        audio.write(reference, data[a:b], rate)
        contour = directory / "reference_f0.npy"
        if args.refresh and contour.exists():
            contour.unlink()
        f0(args, [(reference, contour)], directory)
        values = np.load(contour)
        voiced = values[values > 0]
        if len(voiced) < 30:
            raise ValueError("Reference has too little voiced singing; specify --reference-start for a clean excerpt")
        register_manifest = None
        full_band = None
        if getattr(args, "register_references", False):
            full_rate = int(vocal_info["sample_rate"])
            if full_rate < 44100:
                raise ValueError("Register references need a >=44.1 kHz vocal stem")
            full_band = audio.canonical_at(vocal_source, directory / "source_vocals_full.wav",
                                           rate=full_rate, channels=1)
            register_manifest = registers.extract_registers(full_band, directory / "references")
            mid_name = "mid" if "mid" in register_manifest["registers"] else next(iter(register_manifest["registers"]))
            mid = register_manifest["registers"][mid_name]
            shutil.copy2(mid["prompt"]["path"], reference)
            shutil.copy2(mid["f0"]["path"], contour)
            values = np.load(contour)
            voiced = values[values > 0]
        metadata = None
        if args.prompt_metadata:
            metadata = directory / "prompt_metadata.json"
            validate_metadata(args.prompt_metadata, prompt=True)
            shutil.copy2(args.prompt_metadata, metadata)
            if abs(load(metadata)[0]["time"][1]/1000-(b-a)/rate)>.08:
                raise ValueError("Prompt metadata must align to reference.wav itself, starting at 0ms")
        artifacts = {p.stem: dict(path=str(p.resolve()), sha256=audio.digest(p)) for p in [dry, reference, contour] + ([metadata] if metadata else [])}
        if full_band is not None:
            artifacts["source_vocals_full"] = dict(path=str(full_band.resolve()),
                                                    sha256=audio.digest(full_band))
        if register_manifest is not None:
            register_json = directory / "references" / "registers.json"
            artifacts["registers"] = dict(path=str(register_json.resolve()),
                                          sha256=audio.digest(register_json))
        result.update(status="pending_listening", preparation_status="reference_ready", listening_approved=False,
            reference_candidates=candidates, artifacts=artifacts,
            reference_start_seconds=round(a/rate, 3), reference_seconds=round((b-a)/rate, 3),
            f0_method="librosa_pyin_50hz" if register_manifest else "pyworld_dio_stonemask_50hz",
            voiced_ratio=round(float(np.mean(values > 0)), 4),
            pitch_hz_p10_p50_p90=np.percentile(voiced, [10, 50, 90]).round(2).tolist(),
            capabilities=dict(svc=True, guide_to_svc=True, svs=bool(metadata)), updated_at=now())
        if register_manifest is not None:
            mid_name = "mid" if "mid" in register_manifest["registers"] else next(iter(register_manifest["registers"]))
            result["reference_mode"] = "register"
            result["reference_start_seconds"] = register_manifest["registers"][mid_name]["start_seconds"]
            result["reference_seconds"] = register_manifest["registers"][mid_name]["seconds"]
            result["register_gap"] = register_manifest.get("register_gap")
            result["register_medians_hz"] = {
                name: entry["median_f0_hz"] for name, entry in register_manifest["registers"].items()}
        audio.atomic_json(path, result)
        return result
    except Exception as exc:
        result.update(status="failed", error=str(exc), updated_at=now())
        audio.atomic_json(path, result)
        raise


def validate_metadata(path: Path, *, prompt: bool = False) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(items, list) or not items:
        raise ValueError("SoulX metadata must be a nonempty JSON list")
    end = 0
    for item in items:
        for name in ["time", "phoneme", "duration", "note_pitch", "note_type"]:
            if name not in item:
                raise ValueError(f"Aligned metadata missing {name}; text lyrics alone cannot define sung timing")
        a, b = item["time"]
        if a < end or b <= a or b-a > 25000:
            raise ValueError("Metadata segments must be ordered, nonoverlapping, and at most 25 seconds")
        end = b
        arrays = [str(item[k]).split() for k in ["phoneme", "duration", "note_pitch", "note_type"]]
        if len({len(x) for x in arrays}) != 1 or not arrays[0]:
            raise ValueError("Phoneme/duration/pitch/type token counts must match")
        durations = [float(x) for x in arrays[1]]
        if min(durations) <= 0 or abs(sum(durations) - (b-a)/1000) > .08:
            raise ValueError("Aligned durations must cover the segment time within 80 ms")
    if prompt and len(items) != 1:
        raise ValueError("Prompt metadata must describe exactly the chosen reference excerpt")
    if prompt and items[0]["time"][0]!=0:
        raise ValueError("Prompt metadata must begin at 0ms of reference.wav")
    return items


def remix(instrumental: Path, vocals: Path, output: Path, length: int, gain: float,
          rate: int = audio.RATE) -> dict:
    background,background_rate=audio.read(instrumental)
    voice,voice_rate=audio.read(vocals)
    if background_rate!=rate or voice_rate!=rate:
        raise ValueError(f"Mix inputs must have been canonicalized to {rate} Hz")
    mixed,report=audio.static_mix(background,voice.mean(axis=1),length,gain)
    if mixed.shape[1]==1:
        mixed=np.repeat(mixed,2,axis=1)
    audio.write(output,mixed,rate)
    return report


def archive_score(source:Path,directory:Path,expected_sha256:str)->Path:
    """Freeze the score before inference; external cleanup must not erase its evidence."""
    archived=directory/"target_metadata.json"
    if audio.digest(source)!=expected_sha256:
        raise ValueError("Target metadata changed after task identity was computed")
    if not archived.exists():
        shutil.copy2(source,archived)
    if audio.digest(archived)!=expected_sha256:
        raise ValueError("Archived target metadata does not match inputs.score_sha256")
    return archived.resolve()


def native(args) -> dict:
    """Deliver a YuE2 take as-is; no conversion, no post-processing."""
    if not args.source:
        raise ValueError("native requires --source")
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    info = audio.info(source)
    if info["sample_rate"] < 44100:
        raise ValueError("Native delivery requires a >=44.1 kHz source; this file is 24 kHz")
    title = args.title or source.stem
    out_dir = (args.out_dir or (args.workspace / "native" / safe_id(title))).resolve()
    result = dict(status="planned", pipeline="native_yue2", source=str(source),
                  source_sha256=audio.digest(source), title=title, out_dir=str(out_dir))
    if not args.execute:
        return result
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{title}.wav"
    audio.canonical_at(source, target, rate=info["sample_rate"],
                       channels=min(2, info["channels"]))
    samples, rate = audio.read(target)
    if not np.isfinite(samples).all():
        raise ValueError("Native take contains non-finite samples")
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 1e-4:
        raise ValueError("Native take is silent")
    if args.lyrics_file:
        shutil.copy2(args.lyrics_file, out_dir / "lyrics.txt")
    if args.tags_file:
        shutil.copy2(args.tags_file, out_dir / "tags.txt")
    manifest = dict(result, status="awaiting_listening",
                    output=dict(path=str(target), sha256=audio.digest(target), rate=rate,
                                duration_seconds=round(len(samples) / rate, 3),
                                peak=round(peak, 6), rms=round(rms, 6)),
                    delivery_eligible=True,
                    note="YuE2 native take; no SVC, EQ, compression or limiting.")
    audio.atomic_json(out_dir / "native_manifest.json", manifest)
    return manifest


def conversion(args) -> dict:
    profile_path = args.workspace / "profiles" / safe_id(args.voice_id) / "profile.json"
    singer = load(profile_path)
    if singer.get("preparation_status",singer.get("status")) != "reference_ready":
        raise ValueError(f"Profile is {singer.get('status')}; run profile --execute first")
    for item in singer["artifacts"].values():
        if audio.digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("Singer profile has changed; make a new versioned voice ID")
    kind = "svs" if args.command == "relyric" and args.mode == "svs" else "svc"
    hybrid = args.command == "hybrid" or getattr(args, "preset", "legacy") == "hybrid"
    if hybrid:
        if kind != "svc":
            raise ValueError("hybrid is SVC-only; SVS high-band restoration has not been validated")
        args.chunk_mode = getattr(args, "chunk_mode", None) or "voiced"
        args.f0_method = getattr(args, "f0_method", None) or "pyin"
        args.reference_mode = getattr(args, "reference_mode", None) or "register"
        args.high_band = getattr(args, "high_band", None) or "donor"
        args.mix_rate = getattr(args, "mix_rate", None) or 44100
    else:
        args.chunk_mode = getattr(args, "chunk_mode", None) or "fixed"
        args.f0_method = getattr(args, "f0_method", None) or "dio"
        args.reference_mode = getattr(args, "reference_mode", None) or "single"
        args.high_band = getattr(args, "high_band", None) or "off"
        args.mix_rate = 24000
    source = (args.guide_vocals if args.command == "relyric" and args.mode == "guide" else args.source)
    if source is None:
        raise ValueError("New lyrics require --guide-vocals singing the new lyrics, or --mode svs with aligned --score-json. No old lyric audio will be relabeled as new lyrics.")
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    lyric_hash = audio.digest(args.lyrics_file) if args.command == "relyric" and args.lyrics_file else None
    if args.command == "relyric" and not lyric_hash:
        raise ValueError("relyric requires --lyrics-file to preserve the intended new lyrics")
    spec = dict(schema_version=VERSION, command=args.command, mode=kind, source=str(source),
        source_sha256=audio.digest(source), voice_id=args.voice_id, profile_sha256=audio.digest(profile_path),
        start_seconds=args.start, seconds=args.seconds, chunk_seconds=args.chunk_seconds,
        steps=args.steps, cfg=args.cfg, seed=args.seed, pitch_shift=args.pitch_shift,
        lyrics_sha256=lyric_hash, instrumental_sha256=audio.digest(args.instrumental) if args.instrumental else None,
        vocals_sha256=audio.digest(args.vocals) if args.vocals else None,
        score_sha256=audio.digest(args.score_json) if getattr(args, "score_json", None) else None,
        guide_manifest_sha256=audio.digest(args.guide_manifest) if getattr(args,"guide_manifest",None) else None,
        vocal_gain_db=args.vocal_gain_db, device=args.device, fp32=args.fp32,
        control=args.control if kind=="svs" else None,
        preset="hybrid" if hybrid else "legacy", chunk_mode=args.chunk_mode,
        min_chunk_seconds=getattr(args, "min_chunk_seconds", 6.0),
        max_chunk_seconds=getattr(args, "max_chunk_seconds", 18.0),
        context_seconds=getattr(args, "context_seconds", .25),
        f0_method=args.f0_method, f0_ceil=getattr(args, "f0_ceil", 1600),
        reference_mode=args.reference_mode, high_band=args.high_band,
        mix_rate=args.mix_rate,
        hf_donor_sha256=audio.digest(args.hf_donor_vocals) if getattr(args, "hf_donor_vocals", None) else None)
    # Include backend code + checkpoint identity in cache, even during planned runs.
    for label, path in {"model":args.models/"SoulX-Singer"/("model-svc.pt" if kind=="svc" else "model.pt"),
                        "worker":Path(__file__).with_name("worker.py"), "audio_processor":Path(audio.__file__),
                        "workflow":Path(__file__), "soulx_svc":args.soulx_root/"soulxsinger/models/soulxsinger_svc.py"}.items():
        spec[label+"_sha256"] = audio.digest(path) if path.exists() else None
    spec["soulx_source_sha256"] = audio.fingerprint([(str(p.relative_to(args.soulx_root)),audio.digest(p))
        for folder in ["soulxsinger","cli"] for p in sorted((args.soulx_root/folder).rglob("*.py"))])
    spec["whisper_assets"]=whisper_identity(args.models)
    spec["backend_runtime"]=backend_identity(args.backend_python)
    key = audio.fingerprint(spec)
    directory = args.workspace / "jobs" / (safe_id(args.job_id) if args.job_id else f"{args.command}_{key[:16]}")
    manifest = directory / "manifest.json"
    if manifest.exists():
        previous = load(manifest)
        if previous.get("cache_key") != key:
            raise ValueError("job-id already has different inputs or parameters; use a new job-id")
        if previous.get("status") in {"awaiting_listening", "technical_review_required"}:
            if all(Path(v["path"]).exists() and audio.digest(Path(v["path"]))==v["sha256"] for v in previous["outputs"].values()):
                return previous
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(status="planned", cache_key=key, created_at=now(), manifest=str(manifest.resolve()), inputs=spec,
        listening_approved=False, steps=[], outputs={})
    audio.atomic_json(manifest, result)
    if not args.execute:
        return result
    try:
        score_snapshot=None
        if kind=="svs":
            if not args.score_json:
                raise ValueError("SVS requires --score-json with phoneme/note/duration alignment; raw text is not an aligned score")
            score_snapshot=archive_score(args.score_json.resolve(),directory,spec["score_sha256"])
            result["archived_target_metadata"]=dict(path=str(score_snapshot),sha256=spec["score_sha256"])
        backend = model_spec(args, kind)
        result["status"] = "preparing_audio"
        audio.atomic_json(manifest, result)
        original = directory / ("original"+source.suffix.lower())
        if not original.exists():
            shutil.copy2(source, original)
        if args.command == "relyric" and args.lyrics_file:
            shutil.copy2(args.lyrics_file, directory/"intended_lyrics.txt")
        if getattr(args,"guide_manifest",None):
            shutil.copy2(args.guide_manifest,directory/"guide_source_manifest.json")
        instrumental = args.instrumental
        if args.command == "relyric" and args.mode == "guide":
            vocal_source = source
        elif args.vocals:
            vocal_source = args.vocals
            if abs(audio.info(source)["duration_seconds"]-audio.info(vocal_source)["duration_seconds"])>.08:
                raise ValueError("Source and supplied vocal stem do not share the same full timeline")
        elif kind == "svs":
            vocal_source = source  # melody input is retained for technical A/B duration checks
        else:
            vocal_source, detected = separate(args, source, directory)
            instrumental = instrumental or detected
        target = audio.canonical(vocal_source, directory/"target_vocals.wav", start=args.start, seconds=args.seconds)
        samples, rate = audio.read(target)
        if not len(samples):
            raise ValueError("Selected source excerpt is empty")
        mix_rate = args.mix_rate
        donor = full_target = None
        if hybrid:
            donor_source = getattr(args, "hf_donor_vocals", None) or vocal_source
            donor_info = audio.info(donor_source)
            if donor_info["sample_rate"] < 44100:
                raise ValueError(
                    "Hybrid high-band restoration needs a >=44.1 kHz vocal stem; "
                    "pass --hf-donor-vocals or a --vocals stem from Demucs at 44.1 kHz")
            donor = audio.canonical_at(donor_source, directory/"hf_donor_vocals.wav",
                                       rate=mix_rate, start=args.start, seconds=args.seconds, channels=1)
            full_target = audio.canonical_at(vocal_source, directory/"target_vocals_full.wav",
                                             rate=mix_rate, start=args.start, seconds=args.seconds, channels=1)
        if instrumental:
            if hybrid:
                instrumental = audio.canonical_at(instrumental, directory/"instrumental.wav",
                    start=args.start, seconds=len(samples)/rate, channels=2, rate=mix_rate)
            else:
                instrumental = audio.canonical(instrumental, directory/"instrumental.wav", start=args.start,
                    seconds=len(samples)/rate, channels=2)
        reference = singer["artifacts"]["reference"]["path"]
        prompt_f0 = singer["artifacts"]["reference_f0"]["path"]
        backend.update(prompt_wav=reference, prompt_f0=prompt_f0)
        register_manifest = None
        if hybrid and args.reference_mode == "register":
            artifact = singer["artifacts"].get("registers")
            if not artifact:
                raise ValueError(
                    "Register references are not in this profile; create a v2 profile with "
                    "--register-references or use --reference-mode single")
            register_manifest = load(Path(artifact["path"]))
            backend["prompts"] = [
                {"register": name, "wav": entry["prompt"]["path"], "f0": entry["f0"]["path"]}
                for name, entry in register_manifest["registers"].items()]
            backend["registers_manifest"] = artifact["path"]
        if kind == "svc":
            full_contour = None
            if hybrid:
                contour_path = directory/"target_f0_full.npy"
                f0(args, [(target, contour_path)], directory, method=args.f0_method,
                   f0_floor=65, f0_ceil=args.f0_ceil)
                full_contour = np.load(contour_path)
                entries = audio.voiced_segments(
                    samples, rate, full_contour,
                    min_seconds=args.min_chunk_seconds,
                    max_seconds=args.max_chunk_seconds,
                    context_seconds=args.context_seconds)
            else:
                entries = audio.segments(samples, rate, args.chunk_seconds)
            chunks, parts = [], []
            for entry in entries:
                chunk = directory/f"chunk_{entry['index']:03d}.wav"
                contour = chunk.with_suffix(".f0.npy")
                output = chunk.with_name(chunk.stem+"_converted.wav")
                data = samples[entry["context_start"]:entry["context_end"]]
                audio.write(chunk, data, rate)
                prompt_fields = {}
                if hybrid:
                    # The chunk audio includes the context; the F0 contour must
                    # cover exactly the same samples or the model sings a
                    # shifted melody at every boundary.
                    contour_data = slice_contour(
                        full_contour, entry["context_start"],
                        entry["context_end"] - entry["context_start"], rate)
                    np.save(contour, contour_data)
                    voiced = contour_data[contour_data > 0]
                    median_f0 = float(np.median(voiced)) if len(voiced) else 0.0
                    register = registers.select_register(register_manifest, median_f0) if register_manifest else None
                    if register:
                        reg = register_manifest["registers"][register]
                        prompt_fields = dict(prompt_register=register,
                                             prompt_wav=reg["prompt"]["path"],
                                             prompt_f0=reg["f0"]["path"])
                    else:
                        prompt_fields = dict(prompt_register=None, prompt_wav=reference,
                                             prompt_f0=prompt_f0)
                if np.sqrt(np.mean(data**2)) < .001:
                    audio.write(output, np.zeros(len(data)), rate)
                    entry["silence_passthrough"] = True
                else:
                    prompt_sha = (audio.digest(Path(prompt_fields["prompt_wav"]))
                                  if prompt_fields.get("prompt_wav") else None)
                    item = dict(audio=str(chunk), f0=str(contour), output=str(output),
                        key=audio.fingerprint(dict(job=key, segment=entry,
                                                   prompt=prompt_fields.get("prompt_register"),
                                                   prompt_sha=prompt_sha)))
                    item.update(prompt_fields)
                    chunks.append(item)
                entry["output"] = str(output)
                parts.append((entry, output))
            if not hybrid:
                f0(args, [(Path(c["audio"]), Path(c["f0"])) for c in chunks], directory)
            result.update(status="converting", chunks=entries)
            audio.atomic_json(manifest, result)
            if chunks:
                backend["chunks"] = chunks
                worker(args, "svc", backend, directory)
            generated = audio.merge([(e, audio.read(p)[0].mean(axis=1)) for e,p in parts], len(samples))
        else:
            score = validate_metadata(score_snapshot)
            prompt_meta = singer["artifacts"].get("prompt_metadata")
            if not prompt_meta:
                raise ValueError("SVS needs aligned metadata for reference.wav; supply --prompt-metadata when creating a versioned profile")
            if args.start or args.seconds:
                raise ValueError("SVS score times define its output timeline; crop/edit the score instead of --start/--seconds")
            if abs(score[-1]["time"][1]/1000-len(samples)/rate) > .1:
                raise ValueError("Score and source melody durations differ; align metadata before synthesis")
            backend.update(prompt_metadata=prompt_meta["path"], target_metadata=str(score_snapshot),
                save_dir=str(directory/"svs"), control=args.control)
            worker(args, "svs", backend, directory)
            canonical = audio.canonical(directory/"svs/generated.wav", directory/"svs/generated_pcm.wav")
            generated = audio.read(canonical)[0].mean(axis=1)
            # Upstream can omit trailing rests; explicit padding is allowed only for a <100ms rounding tail.
            if abs(len(generated)-len(samples)) <= rate*.1:
                generated = np.pad(generated, (0,max(0,len(samples)-len(generated))))[:len(samples)]
        raw = directory/"converted_raw_vocals.wav"
        audio.write(raw,generated,rate)
        qa = audio.technical_qa(samples.mean(axis=1), generated, rate)
        delivery_rate = rate
        hf_report = None
        highnote = None
        if hybrid:
            source_full, _ = audio.read(full_target)
            source_full = source_full.mean(axis=1)
            donor_data, donor_rate = audio.read(donor)
            donor_data = donor_data.mean(axis=1)
            restored, hf_report = band_restore.restore_high_band(
                generated, rate, donor_data, donor_rate, out_rate=mix_rate,
                cutoff_hz=args.hf_crossover)
            matched, level_report = audio.match_vocal_level(source_full, restored, mix_rate)
            delivery_rate = mix_rate
            dry = directory/"converted_vocals.wav"
            audio.write(dry, matched, delivery_rate)
            if getattr(args, "highnote_qa", False):
                try:
                    highnote = highnote_qa.analyze(
                        matched, delivery_rate, source=source_full, source_rate=mix_rate)
                    highnote.update(file=str(dry.resolve()), source=str(Path(full_target).resolve()))
                    audio.atomic_json(directory/"highnote_qa.json", highnote)
                except Exception as error:  # evidence failure must not hide the rendered audio
                    highnote = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                    audio.atomic_json(directory/"highnote_qa.json", highnote)
        else:
            matched, level_report = audio.match_vocal_level(samples.mean(axis=1), generated, rate)
            dry = directory/"converted_vocals.wav"
            audio.write(dry, matched, delivery_rate)
        outputs = {"raw_vocals":dict(path=str(raw.resolve()),sha256=audio.digest(raw)),
            "vocals":dict(path=str(dry.resolve()), sha256=audio.digest(dry))}
        mix_report=None
        if instrumental:
            mix = directory/"review_mix.wav"
            mix_report=remix(instrumental, dry, mix, len(matched), args.vocal_gain_db, rate=delivery_rate)
            outputs["mix"] = dict(path=str(mix.resolve()), sha256=audio.digest(mix))
        audio.atomic_json(directory/"technical_qa.json", qa)
        delivery_gate = dict(deliverable=bool(hybrid and delivery_rate >= 44100), reasons=[])
        if hybrid:
            if delivery_rate < 44100:
                delivery_gate["reasons"].append("delivery_rate_below_44k")
            if highnote is None:
                delivery_gate["deliverable"] = False
                delivery_gate["reasons"].append("highnote_qa_not_run")
            elif highnote.get("status") == "failed":
                delivery_gate["deliverable"] = False
                delivery_gate["reasons"].append("highnote_qa_failed")
            else:
                for flag in ("band_limited_24k", "pitch_tracking_mismatch"):
                    if flag in highnote.get("flags", []):
                        delivery_gate["deliverable"] = False
                        delivery_gate["reasons"].append(flag)
        else:
            delivery_gate = dict(deliverable=False, reasons=["legacy_24k_preset"])
        warnings = list(qa["warnings"])
        if highnote and "band_limited_24k" in highnote.get("flags", []):
            warnings.append("High-band restoration still looks band-limited")
        result.update(status="technical_review_required" if warnings else "awaiting_listening",
            outputs=outputs, technical_qa=qa, updated_at=now(), model_sha256=backend["model_sha256"],
            vocal_level_match=level_report,mix_gain=mix_report,
            reference_path=reference, pipeline=dict(version=VERSION, preset="hybrid" if hybrid else "legacy"),
            high_band=hf_report, highnote_qa=highnote, delivery_gate=delivery_gate,
            review_items=["像同一个女声吗", "高音与转音是否还有机械感", "逐句检查漏字/错字", "段落接缝是否突变", "伴奏与人声节奏是否对齐"])
        audio.atomic_json(manifest, result)
        return result
    except Exception as exc:
        result.update(status="failed", error=str(exc), updated_at=now())
        audio.atomic_json(manifest, result)
        raise


def doctor(args) -> dict:
    paths = dict(backend_python=args.backend_python, audio_python=args.audio_python,
        soulx_root=args.soulx_root, svc_model=args.models/"SoulX-Singer/model-svc.pt",
        svs_model=args.models/"SoulX-Singer/model.pt", whisper=args.models/"whisper-base",
        whisper_config=args.models/"whisper-base/config.json",
        whisper_preprocessor=args.models/"whisper-base/preprocessor_config.json",
        whisper_weights=args.models/"whisper-base/model.safetensors",
        soulx_config=args.soulx_root/"soulxsinger/config/soulxsinger.yaml",
        ffmpeg=Path(shutil.which("ffmpeg") or "__missing__"))
    report = dict(paths={k:dict(path=str(v), exists=v.exists()) for k,v in paths.items()},
        gpu_inference_run=False, status="ready_to_probe" if all(paths[k].exists() for k in ["backend_python","audio_python","soulx_root","soulx_config","svc_model","whisper_config","whisper_preprocessor","whisper_weights","ffmpeg"]) else "dependencies_missing")
    for kind in ["svc","svs"]:
        checkpoint=paths[kind+"_model"]
        report["paths"][kind+"_model"]["complete_size"]=checkpoint.is_file() and checkpoint.stat().st_size==OFFICIAL_SIZES[kind]
    if not report["paths"]["svc_model"]["complete_size"]:
        report["status"]="dependencies_missing"
    whisper_weights=paths["whisper_weights"]
    report["paths"]["whisper_weights"]["complete_size"]=whisper_weights.is_file() and whisper_weights.stat().st_size==ASSETS["openai/whisper-base"]["files"]["model.safetensors"]["size"]
    if not report["paths"]["whisper_weights"]["complete_size"]:
        report["status"]="dependencies_missing"
    if args.probe and args.backend_python.is_file():
        temp = args.workspace/"doctor"
        worker(args, "probe", {}, temp)
        lines=(temp/"probe.log").read_text(errors="replace").splitlines()
        report["backend_imports"] = json.loads(next(x for x in reversed(lines) if x.startswith("{")))
        if any(isinstance(v,dict) and "error" in v for v in report["backend_imports"].values()):
            report["status"]="dependencies_missing"
        if args.audio_python.is_file():
            worker(args,"probe",dict(modules=["torch","demucs","pyworld","soundfile","numpy"]),temp/"audio",audio_backend=True)
            lines=(temp/"audio/probe.log").read_text(errors="replace").splitlines()
            report["audio_imports"]=json.loads(next(x for x in reversed(lines) if x.startswith("{")))
            if any(isinstance(v,dict) and "error" in v for v in report["audio_imports"].values()):
                report["status"]="dependencies_missing"
    return report


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Fixed singer: reference profile, SVC covers and aligned/guide re-lyrics")
    sub = cli.add_subparsers(dest="command", required=True)
    for name in ["doctor", "profile", "cover", "relyric", "native", "hybrid"]:
        p = sub.add_parser(name)
        p.add_argument("--workspace", type=Path, default=PROJECT/"data/voice_studio")
        p.add_argument("--backend-python", type=Path, default=PROJECT/"vendor/voice_studio/.venv/Scripts/python.exe")
        p.add_argument("--audio-python", type=Path, default=PROJECT/"tools/vocal_repair/.venv/Scripts/python.exe")
        p.add_argument("--soulx-root", type=Path, default=PROJECT/"vendor/voice_studio/SoulX-Singer")
        p.add_argument("--models", type=Path, default=PROJECT/"models/voice_studio")
        if name == "doctor":
            p.add_argument("--probe", action="store_true")
            continue
        p.add_argument("--voice-id", default="iron_rain_female_v1")
        p.add_argument("--source", type=Path, default=DEFAULT_SOURCE if name=="profile" else None)
        p.add_argument("--vocals", type=Path, help="Existing dry vocal stem; avoids another separation")
        p.add_argument("--execute", action="store_true", help="Perform audio preparation/inference; omission writes a plan only")
        p.add_argument("--separation-device", default="cpu", choices=["cpu","cuda"])
        if name == "native":
            p.add_argument("--out-dir", type=Path)
            p.add_argument("--title")
            p.add_argument("--lyrics-file", type=Path)
            p.add_argument("--tags-file", type=Path)
            continue
        if name == "profile":
            p.add_argument("--reference-start", type=float)
            p.add_argument("--reference-seconds", type=float, default=8)
            p.add_argument("--prompt-metadata", type=Path)
            p.add_argument("--source-manifest",type=Path,help="Parent original hash and crop interval provenance, archived with the profile")
            p.add_argument("--parent-source",type=Path,help="Archive the full parent original as well when --source is a cropped excerpt")
            p.add_argument("--refresh", action="store_true")
            p.add_argument("--register-references", action="store_true",
                           help="Extract low/mid/high reference clips from the full-band stem (new voice id required)")
            continue
        p.add_argument("--job-id")
        p.add_argument("--instrumental", type=Path)
        p.add_argument("--start", type=float, default=0)
        p.add_argument("--seconds", type=float, default=30)
        p.add_argument("--chunk-seconds", type=float, default=12)
        p.add_argument("--device", default="cuda")
        p.add_argument("--fp32", action="store_true")
        p.add_argument("--steps", type=int, default=32)
        p.add_argument("--cfg", type=float, default=3)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--pitch-shift", type=int, default=0, help="Explicit semitones; never changes key automatically")
        p.add_argument("--vocal-gain-db", type=float, default=0)
        p.add_argument("--preset", choices=["legacy", "hybrid"], default="legacy")
        p.add_argument("--chunk-mode", choices=["fixed", "voiced"], default=None)
        p.add_argument("--min-chunk-seconds", type=float, default=6)
        p.add_argument("--max-chunk-seconds", type=float, default=18)
        p.add_argument("--context-seconds", type=float, default=.25)
        p.add_argument("--f0-method", choices=["dio", "pyin"], default=None)
        p.add_argument("--f0-ceil", type=float, default=1600)
        p.add_argument("--reference-mode", choices=["single", "register"], default=None)
        p.add_argument("--high-band", choices=["off", "donor"], default=None)
        p.add_argument("--hf-crossover", type=float, default=11000)
        p.add_argument("--hf-donor-vocals", type=Path)
        p.add_argument("--mix-rate", type=int, default=44100, choices=[24000, 44100, 48000])
        p.add_argument("--highnote-qa", action="store_true",
                       help="Run the high-note evidence pass on the rendered vocal")
        if name == "relyric":
            p.add_argument("--mode", choices=["guide","svs"], default="guide")
            p.add_argument("--guide-vocals", "--guide", dest="guide_vocals", type=Path, help="Dry sung guide already containing the intended new lyrics")
            p.add_argument("--guide-manifest",type=Path,help="Optional generator provenance JSON retained with the new-lyric guide")
            p.add_argument("--lyrics-file", type=Path)
            p.add_argument("--score-json", type=Path)
            p.add_argument("--control", choices=["melody","score"], default="score")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for key in ["workspace","backend_python","audio_python","soulx_root","models"]:
        setattr(args, key, getattr(args,key).resolve())
    try:
        if getattr(args,"start",0)<0 or getattr(args,"seconds",0)<0:
            raise ValueError("start and seconds must be nonnegative; seconds=0 means full song")
        if getattr(args,"steps",32)<1 or getattr(args,"cfg",3)<0:
            raise ValueError("Invalid inference settings")
        value = (doctor(args) if args.command=="doctor"
                 else profile(args) if args.command=="profile"
                 else native(args) if args.command=="native"
                 else conversion(args))
        print(json.dumps(value, ensure_ascii=False, indent=2))
        if args.command=="doctor" and value["status"]=="dependencies_missing":
            return 1
        return 0
    except Exception as exc:
        print(json.dumps(dict(status="failed", error=str(exc)), ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
