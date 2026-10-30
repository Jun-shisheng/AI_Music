"""Re-render selected SVC chunks of a completed fixed-voice cover and reassemble.

Only the chosen 12-second chunks are sent through SoulX-SVC with new seed/steps;
every other chunk is reused byte-for-byte from the parent job. The merge, level
matching, technical QA and mix use the same helpers as the normal workflow, so
the result is directly comparable to the parent take.

The parent job is never modified. Outputs go to a new directory with a manifest
recording the parent, selected chunks, parameters and hashes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.voice_studio import audio  # noqa: E402

JOBS = ROOT / "data/voice_studio/jobs"
BACKEND = ROOT / "vendor/voice_studio/.venv/Scripts/python.exe"
WORKER = ROOT / "tools/voice_studio/worker.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_chunks(value: str, chunks: list[dict], times: str | None) -> list[int]:
    if value and times:
        raise SystemExit("Use either --chunks or --times, not both")
    if value:
        return sorted({int(x) for x in value.replace(",", " ").split()})
    if times:
        ranges = []
        for part in times.replace(";", ",").split(","):
            if not part.strip():
                continue
            a, b = part.split("-")
            ranges.append((float(a), float(b)))
        selected = []
        for chunk in chunks:
            start = chunk["start"] / audio.RATE
            end = chunk["end"] / audio.RATE
            if any(start < hi and end > lo for lo, hi in ranges):
                selected.append(chunk["index"])
        return sorted(set(selected))
    raise SystemExit("Select chunks with --chunks or --times")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--chunks")
    parser.add_argument("--times", help='Seconds, e.g. "6-9,32-35"')
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="retake")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    job = JOBS / args.job_id
    manifest = load(job / "manifest.json")
    spec = load(job / "svc_spec.json")
    if manifest.get("status") not in {"awaiting_listening", "technical_review_required"}:
        raise SystemExit(f"Parent job status is {manifest.get('status')!r}; refusing")

    chunks = manifest["chunks"]
    selected = parse_chunks(args.chunks or "", chunks, args.times)
    known = {c["index"] for c in chunks}
    missing = [i for i in selected if i not in known]
    if missing or not selected:
        raise SystemExit(f"Unknown/empty chunk selection: {missing or selected}")

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    retake_spec = dict(spec)
    retake_spec["seed"] = args.seed
    retake_spec["steps"] = args.steps
    retake_spec["chunks"] = []
    for chunk in chunks:
        if chunk["index"] not in selected:
            continue
        index = chunk["index"]
        chunk_wav = job / f"chunk_{index:03d}.wav"
        chunk_f0 = job / f"chunk_{index:03d}.f0.npy"
        if chunk.get("silence_passthrough") or not chunk_wav.is_file() or not chunk_f0.is_file():
            continue
        retake_spec["chunks"].append(dict(
            audio=str(chunk_wav), f0=str(chunk_f0),
            output=str(out / f"chunk_{index:03d}_converted.wav"),
            key=audio.fingerprint(dict(job=args.job_id, chunk=index,
                                       seed=args.seed, steps=args.steps)),
        ))
    spec_path = out / "svc_retake_spec.json"
    audio.atomic_json(spec_path, retake_spec)
    if args.plan_only:
        print(json.dumps(dict(selected=selected, spec=str(spec_path),
                              chunks=retake_spec["chunks"]), ensure_ascii=False, indent=1))
        return 0

    log_path = out / "retake.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        code = subprocess.run(
            [str(BACKEND), str(WORKER), "svc", "--spec", str(spec_path)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        ).returncode
    if code != 0:
        raise SystemExit(f"Worker failed with exit {code}; see {log_path}")

    outputs = {}
    for item in retake_spec["chunks"]:
        path = Path(item["output"])
        expected = None
        for chunk in chunks:
            if chunk["index"] == int(path.stem.split("_")[1]):
                expected = chunk["context_end"] - chunk["context_start"]
        data, rate = audio.read(path)
        if expected is not None and len(data) != expected:
            raise SystemExit(f"Chunk length mismatch: {path} {len(data)} != {expected}")
        outputs[path.stem] = dict(path=str(path), sha256=audio.digest(path))

    target, target_rate = audio.read(job / "target_vocals.wav")
    if target_rate != audio.RATE:
        raise SystemExit("Target vocals are not 24 kHz")
    rendered = {int(Path(item["output"]).stem.split("_")[1]) for item in retake_spec["chunks"]}
    parts = []
    for chunk in chunks:
        path = out / f"chunk_{chunk['index']:03d}_converted.wav" if chunk["index"] in rendered else Path(chunk["output"])
        data, rate = audio.read(path)
        if rate != audio.RATE:
            raise SystemExit(f"Chunk sample rate mismatch: {path}")
        parts.append((chunk, data.mean(axis=1)))
    generated = audio.merge(parts, len(target))
    source = target.mean(axis=1)
    matched, level_report = audio.match_vocal_level(source, generated, audio.RATE)
    qa = audio.technical_qa(source, generated, audio.RATE)

    raw_path = out / "converted_raw_vocals.wav"
    dry_path = out / "converted_vocals.wav"
    audio.write(raw_path, generated, audio.RATE)
    audio.write(dry_path, matched, audio.RATE)
    instrumental, inst_rate = audio.read(job / "instrumental.wav")
    if inst_rate != audio.RATE:
        raise SystemExit("Instrumental is not 24 kHz")
    mixed, mix_report = audio.static_mix(instrumental, matched, len(target), 0)
    if mixed.shape[1] == 1:
        mixed = np.repeat(mixed, 2, axis=1)
    mix_path = out / "review_mix.wav"
    audio.write(mix_path, mixed, audio.RATE)

    summary = dict(
        schema_version=1, kind="chunk_retake", label=args.label, created_at=now(),
        parent_job=args.job_id,
        parent_manifest_sha256=audio.digest(job / "manifest.json"),
        parent_take_status=manifest.get("status"),
        seed=args.seed, steps=args.steps, cfg=spec.get("cfg"),
        selected_chunks=[
            dict(index=c["index"], start_seconds=round(c["start"] / audio.RATE, 3),
                 end_seconds=round(c["end"] / audio.RATE, 3))
            for c in chunks if c["index"] in selected
        ],
        unchanged_chunks=len(chunks) - len(selected),
        outputs=dict(raw=str(raw_path), dry=str(dry_path), mix=str(mix_path),
                     dry_sha256=audio.digest(dry_path), mix_sha256=audio.digest(mix_path)),
        chunk_outputs=outputs,
        technical_qa=qa, vocal_level_match=level_report, mix_gain=mix_report,
        inference_performed=True, listening_approved=False,
        note="Only selected chunks were re-inferred; other chunks are byte-identical to the parent.",
    )
    audio.atomic_json(out / "retake_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
