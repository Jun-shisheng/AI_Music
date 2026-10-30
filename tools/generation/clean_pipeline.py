"""Source-clean HeartMuLa generation with ZERO post-processing repair.

Design principles (redesigned 2026-08-15; retuned for the 12 GB desktop
2026-10-09):

1. SOURCE QUALITY FIRST: official HeartMuLa weights, no frame repair / EQ /
   humanize / de-esser chain ever touches the audio.
2. MULTI-CANDIDATE DIVERSITY: the same lyrics/tags are generated several times
   with different seeds and sampling parameters.
3. ZERO TOUCH POST: the only allowed step is loudness normalization (a linear
   gain -- it does not change the spectrum).
4. LISTENING GATE: the user picks the take. Objective checks only catch broken
   files (duration, peak, clipping, near-silence) -- they never reject on
   spectral proxies (zcr cannot separate sibilants from noise).
5. RIGHT-SIZED KV CACHE: the backbone preallocates its cache for `max_seq_len`
   (default 8192) regardless of song length. Right-sizing it to prompt+frames
   frees ~2.4 GB at guidance batch 2, which is what makes cfg>1.0 affordable.
6. GUIDANCE IS ON BY DEFAULT: cfg_scale=1.0 disables classifier-free guidance
   entirely (batch 1). That was FORCED on the old 8 GB laptop, and every
   previously delivered song was generated that way. Measured on this machine,
   cfg=2.0 cuts dropouts 6 -> 1 and sample-level clicks 854 -> 469 versus
   cfg=1.0; cfg=3.0 over-drives (4927 clicks). See tools/generation/audio_qa.py.

12 GB measurements (RTX 3060, codec on GPU, right-sized cache):
  cfg=1.0 30 s: peak 8.5 GB token / 10.7 GB decode, 1.6 min end-to-end
  cfg=2.0 30 s: peak 9.1 GB token / 10.7 GB decode, 1.6 min end-to-end
  cfg=2.0 60 s: peak 9.4 GB token / 10.7 GB decode, 2.5 min end-to-end
  token generation runs ~10 it/s at every cfg -- guidance is free once the
  cache is right-sized.

Usage:
  python tools/generation/clean_pipeline.py --lyrics outputs/candidates/x.txt \
      --tags outputs/candidates/x_tags.txt --title silver_rain \
      --candidates 4 --seconds 30
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loudness  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEART_ROOT = PROJECT_ROOT / "vendor" / "heartlib"
PYTHON = PROJECT_ROOT / "vendor" / "ACE-Step-1.5" / ".venv" / "Scripts" / "python.exe"
MODEL_ROOT = PROJECT_ROOT / "models" / "heartmula"

# One in-process generation: load mula, right-size the KV cache, sample frames,
# decode with the codec, save 48 kHz WAV. Emits a JSON result line.
_RUNNER = r'''
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("project_root", type=Path)
ap.add_argument("model_root", type=Path)
ap.add_argument("--lyrics", type=Path, required=True)
ap.add_argument("--tags", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--seconds", type=int, required=True)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--topk", type=int, default=50)
ap.add_argument("--cfg", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--codec-device", default="cuda")
ap.add_argument("--seq-margin", type=int, default=384)
a = ap.parse_args()

sys.path.insert(0, str(a.project_root / "vendor" / "heartlib" / "src"))
import numpy as np
import torch
from heartlib import HeartMuLaGenPipeline

random.seed(a.seed); np.random.seed(a.seed)
torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)

lyrics = a.lyrics.read_text(encoding="utf-8")
tags = a.tags.read_text(encoding="utf-8")

t0 = time.time()
pipe = HeartMuLaGenPipeline.from_pretrained(
    str(a.model_root),
    device={"mula": torch.device("cuda"), "codec": torch.device(a.codec_device)},
    dtype={"mula": torch.bfloat16, "codec": torch.float32},
    version="3B",
    lazy_load=True,
)
pre = pipe.preprocess({"lyrics": lyrics, "tags": tags}, cfg_scale=a.cfg)
prompt_len = int(pre["pos"].shape[-1])
needed = prompt_len + a.seconds * 1000 // 80 + a.seq_margin
seq_len = max(1024, -(-needed // 512) * 512)
pipe.mula.backbone.max_seq_len = seq_len
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    out = pipe._forward(pre, max_audio_length_ms=a.seconds * 1000,
                        temperature=a.temperature, topk=a.topk, cfg_scale=a.cfg)
frames = out["frames"]
peak_tok = torch.cuda.max_memory_allocated() / 1e9
t1 = time.time()

import torchaudio
wav = pipe.codec.detokenize(frames.to(pipe.codec.device))
a.out.parent.mkdir(parents=True, exist_ok=True)
torchaudio.save(str(a.out), wav.to(torch.float32).cpu(), 48000)
print(json.dumps({
    "ok": True, "prompt_len": prompt_len, "seq_len": seq_len,
    "frames": int(frames.shape[-1]), "peak_tok_gb": round(peak_tok, 2),
    "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    "minutes": round((time.time() - t0) / 60, 2),
}), flush=True)
'''


@dataclass
class Candidate:
    name: str
    path: Path
    seed: int
    temperature: float
    topk: int
    cfg_scale: float
    duration_s: float
    peak: float
    clipping_ratio: float
    ok: bool
    reason: str = ""
    # loudness report (see tools/generation/loudness.py) -- recorded so that a
    # silent gain cap can be spotted from the manifest instead of going unnoticed
    rms_in: float = 0.0
    gain_db: float = 0.0
    boost_capped: bool = False
    peak_limited: bool = False


def check_audio(path: Path, min_duration_s: float) -> dict:
    """Lightweight objective check: duration, peak, clipping, near-silence."""
    import numpy as np
    import soundfile as sf

    info = sf.info(path)
    y, sr = sf.read(path, always_2d=True)
    mono = y.mean(axis=1)
    peak = float(np.max(np.abs(mono)))
    clip = float(np.mean(np.abs(mono) > 0.999))
    rms = float(np.sqrt(np.mean(mono**2)))
    duration = float(info.duration)
    ok = True
    reasons = []
    if duration < min_duration_s * 0.5:
        ok, reasons = False, [f"too short: {duration:.1f}s"]
    if peak < 0.02:
        ok, reasons = False, [f"near-silent: peak={peak:.4f}"]
    if clip > 0.001:
        ok, reasons = False, [f"clipping: {clip*100:.1f}%"]
    if rms < 0.005:
        ok, reasons = False, [f"too quiet: rms={rms:.4f}"]
    return {"duration_s": duration, "peak": peak, "clipping_ratio": clip, "rms": rms,
            "ok": ok, "reasons": reasons}


def generate_one(lyrics: Path, tags: Path, output: Path, seconds: int,
                 seed: int, temperature: float, topk: int, cfg_scale: float,
                 codec_device: str = "cuda", seq_margin: int = 384) -> dict:
    """Run one HeartMuLa generation via the in-process runner.

    Retries once with the codec on CPU if the GPU decode runs out of memory, so
    a long song still completes instead of failing the whole batch.
    """
    runner = output.parent / f"_runner_{output.stem}.py"
    runner.write_text(_RUNNER, encoding="utf-8")
    try:
        for device in ([codec_device] if codec_device == "cpu" else [codec_device, "cpu"]):
            cmd = [str(PYTHON), "-I", str(runner), str(PROJECT_ROOT), str(MODEL_ROOT),
                   "--lyrics", str(lyrics.resolve()), "--tags", str(tags.resolve()),
                   "--out", str(output.resolve()), "--seconds", str(seconds),
                   "--temperature", str(temperature), "--topk", str(topk),
                   "--cfg", str(cfg_scale), "--seed", str(seed),
                   "--codec-device", device, "--seq-margin", str(seq_margin)]
            proc = subprocess.run(cmd, cwd=HEART_ROOT, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            if proc.returncode == 0:
                for line in reversed(proc.stdout.splitlines()):
                    if line.startswith("{"):
                        return json.loads(line)
                return {}
            log = (proc.stdout or "") + (proc.stderr or "")
            oom = "out of memory" in log.lower() or "OutOfMemoryError" in log
            if device != "cpu" and oom:
                print(f"    GPU decode OOM at codec; retrying with codec on CPU", flush=True)
                continue
            raise RuntimeError(log[-1200:])
        raise RuntimeError("generation failed on every codec device")
    finally:
        runner.unlink(missing_ok=True)


def normalize_loudness(src: Path, dst: Path) -> dict:
    """Linear loudness normalization to ~-16 dBFS; spectrum-preserving.

    Thin wrapper over `loudness.normalize_loudness` (the single implementation --
    the previous copy-pasted version capped the boost at +12 dB, which left 17 of
    the 20 delivered songs below target). Returns the report so callers can record
    the raw level and the applied gain.
    """
    return loudness.normalize_loudness(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Source-clean multi-candidate generation.")
    parser.add_argument("--lyrics", type=Path, required=True)
    parser.add_argument("--tags", type=Path, required=True)
    parser.add_argument("--title", default="candidate")
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "candidates")
    parser.add_argument("--base-cfg", type=float, default=2.0,
                        help="CFG scale for all candidates (default 2.0 = library-style "
                             "guidance). Measured best on 12 GB: fewest dropouts and "
                             "clicks. 1.0 disables guidance (the old 8 GB fallback).")
    parser.add_argument("--codec-device", default="cuda",
                        help="'cuda' (fast, default on 12 GB) or 'cpu'. Automatically "
                             "falls back to CPU if the GPU decode runs out of memory.")
    parser.add_argument("--seq-margin", type=int, default=384,
                        help="Extra KV-cache positions above prompt+frames.")
    parser.add_argument("--full-song", action="store_true",
                        help="Deprecated: kept for CLI compatibility. Long songs now "
                             "use the same single-pass runner.")
    parser.add_argument("--base-seed", type=int, default=100,
                        help="Seed for the first candidate; subsequent candidates add 7.")
    args = parser.parse_args()

    if args.full_song:
        print("[note] --full-song is deprecated; all lengths use the single-pass runner.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / f"{args.title}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Candidate diversity comes from temperature/topk/seed. cfg stays at base so
    # every take is generated at the same guidance strength.
    configs = []
    for i in range(args.candidates):
        configs.append({
            "name": f"{args.title}_c{i+1}", "seed": args.base_seed + i * 7,
            "temperature": [0.8, 1.0, 1.2, 0.9][i % 4],
            "topk": [30, 50, 70, 40][i % 4],
            "cfg_scale": args.base_cfg,
        })

    print(f"Generating {len(configs)} candidates into {run_dir} "
          f"(cfg={args.base_cfg}, codec={args.codec_device})")
    results: list[Candidate] = []
    for c in configs:
        raw = run_dir / f"{c['name']}_raw.wav"
        final = run_dir / f"{c['name']}.wav"
        print(f"\n--- {c['name']} seed={c['seed']} cfg={c['cfg_scale']} "
              f"temp={c['temperature']} topk={c['topk']} ---", flush=True)
        try:
            info = generate_one(args.lyrics, args.tags, raw, args.seconds,
                                c["seed"], c["temperature"], c["topk"], c["cfg_scale"],
                                codec_device=args.codec_device, seq_margin=args.seq_margin)
            print(f"  generated {info.get('frames')} frames, seq={info.get('seq_len')}, "
                  f"peak={info.get('peak_gb')}GB in {info.get('minutes')}min", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  generation failed: {exc}")
            results.append(Candidate(name=c["name"], path=raw, ok=False, reason=str(exc),
                                     seed=c["seed"], temperature=c["temperature"],
                                     topk=c["topk"], cfg_scale=c["cfg_scale"],
                                     duration_s=0.0, peak=0.0, clipping_ratio=0.0))
            continue
        check = check_audio(raw, args.seconds)
        if not check["ok"]:
            print(f"  rejected by objective check: {check['reasons']}")
            results.append(Candidate(name=c["name"], path=raw, ok=False, reason=check["reasons"][0],
                                     seed=c["seed"], temperature=c["temperature"],
                                     topk=c["topk"], cfg_scale=c["cfg_scale"],
                                     duration_s=check["duration_s"], peak=check["peak"],
                                     clipping_ratio=check["clipping_ratio"]))
            continue
        level = normalize_loudness(raw, final)
        raw.unlink(missing_ok=True)
        flags = []
        if level["boost_capped"]:
            flags.append(f"boost capped at {level['gain_db']:.1f} dB")
        if level["peak_limited"]:
            flags.append(f"peak-limited ({level['shortfall_db']:.1f} dB short)")
        note = f"; {'; '.join(flags)}" if flags else ""
        print(f"  OK -> {final} ({check['duration_s']:.1f}s peak={check['peak']:.2f} "
              f"rms {level['rms_in_dbfs']:.1f} -> {level['rms_out_dbfs']:.1f} dBFS{note}")
        results.append(Candidate(name=c["name"], path=final, ok=True,
                                 seed=c["seed"], temperature=c["temperature"],
                                 topk=c["topk"], cfg_scale=c["cfg_scale"],
                                 duration_s=check["duration_s"], peak=check["peak"],
                                 clipping_ratio=check["clipping_ratio"],
                                 rms_in=level["rms_in"], gain_db=level["gain_db"],
                                 boost_capped=level["boost_capped"],
                                 peak_limited=level["peak_limited"]))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "title": args.title,
        "lyrics": str(args.lyrics),
        "tags": str(args.tags),
        "base_cfg": args.base_cfg,
        "codec_device": args.codec_device,
        "candidates": [{"name": r.name, "path": str(r.path), "ok": r.ok,
                        "seed": r.seed, "temperature": r.temperature, "topk": r.topk,
                        "cfg_scale": r.cfg_scale, "duration_s": r.duration_s,
                        "peak": r.peak, "clipping_ratio": r.clipping_ratio,
                        "rms_in": r.rms_in, "gain_db": r.gain_db,
                        "boost_capped": r.boost_capped, "peak_limited": r.peak_limited,
                        "reason": r.reason} for r in results],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ok_count = sum(1 for r in results if r.ok)
    print(f"\nPassed objective check: {ok_count}/{len(results)}")
    for r in results:
        status = "OK  " if r.ok else "FAIL"
        print(f"  [{status}] {r.name}  cfg={r.cfg_scale} {r.path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
