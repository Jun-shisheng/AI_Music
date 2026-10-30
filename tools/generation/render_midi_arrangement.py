"""Render a controlled MIDI arrangement with FluidSynth and mix a fixed vocal.

The MIDI is rendered offline at 44.1 kHz, trimmed/padded to the vocal's exact
resampled length, brought to a constant target RMS, and summed with the vocal.
Only a short instrumental tail fade and one global peak-headroom gain are
applied; the vocal itself is not processed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import gcd
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLUIDSYNTH = Path("C:/tools/fluidsynth/bin/fluidsynth.exe")
DEFAULT_SOUNDFONT = ROOT / "models/soundfonts/GeneralUser-GS.sf2"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--vocal", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--soundfont", type=Path, default=DEFAULT_SOUNDFONT)
    parser.add_argument("--fluidsynth", type=Path, default=DEFAULT_FLUIDSYNTH)
    parser.add_argument("--target-rms", type=float, default=0.07)
    parser.add_argument("--gain", type=float, default=0.7, help="FluidSynth synth gain")
    parser.add_argument("--tail-fade-ms", type=float, default=50)
    parser.add_argument("--mix-rate", type=int, default=44100, choices=[44100, 48000],
                        help="Output rate; the vocal is resampled, never time-stretched")
    args = parser.parse_args()

    fluidsynth = args.fluidsynth if args.fluidsynth.is_file() else Path(shutil.which("fluidsynth") or "")
    if not fluidsynth.is_file():
        raise SystemExit("FluidSynth not found")
    if not args.soundfont.is_file():
        raise SystemExit(f"SoundFont not found: {args.soundfont}")

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or args.midi.stem
    raw = out / f"{name}_raw.wav"
    mix_rate = int(args.mix_rate)
    subprocess.run([str(fluidsynth), "-ni", "-F", str(raw), "-T", "wav", "-r", str(mix_rate),
                    "-g", str(args.gain), str(args.soundfont.resolve()), str(args.midi.resolve())],
                   check=True, capture_output=True)

    vocal, vocal_rate = sf.read(str(args.vocal), dtype="float64")
    if vocal.ndim > 1:
        vocal = vocal.mean(axis=1)
    if vocal_rate != mix_rate:
        common = gcd(int(vocal_rate), mix_rate)
        vocal = resample_poly(vocal, mix_rate // common, int(vocal_rate) // common)
    up = vocal.astype(np.float64)
    n = len(up)

    instrumental, inst_rate = sf.read(str(raw), dtype="float64", always_2d=True)
    if inst_rate != mix_rate:
        raise SystemExit(f"Expected a {mix_rate} Hz render, got {inst_rate}")
    instrumental = instrumental[:n]
    if len(instrumental) < n:
        instrumental = np.pad(instrumental, ((0, n - len(instrumental)), (0, 0)))
    raw_rms = float(np.sqrt(np.mean(instrumental ** 2)))
    gain = args.target_rms / raw_rms if raw_rms else 0.0
    instrumental = instrumental * gain
    fade = min(int(args.tail_fade_ms / 1000 * mix_rate), len(instrumental))
    if fade:
        instrumental[-fade:] *= np.linspace(1, 0, fade, endpoint=False)[:, None]

    mix = instrumental + up[:, None]
    peak_before = float(np.max(np.abs(mix))) if len(mix) else 0.0
    global_gain = min(1.0, 0.95 / peak_before) if peak_before else 1.0
    mix = mix * global_gain

    suffix = "44k" if mix_rate == 44100 else "48k"
    inst_path = out / f"{name}_instrumental_{suffix}.wav"
    mix_path = out / f"{name}_mix_{suffix}.wav"
    sf.write(str(inst_path), instrumental, mix_rate, subtype="PCM_16")
    sf.write(str(mix_path), mix, mix_rate, subtype="PCM_16")

    report = {
        "schema": 1, "kind": "controlled_midi_render_mix", "name": name,
        "status": "rendered_awaiting_listening",
        "renderer": {"name": "FluidSynth", "path": str(fluidsynth), "gain": args.gain},
        "soundfont": {"path": str(args.soundfont), "sha256": sha(args.soundfont)},
        "midi": {"path": str(args.midi), "sha256": sha(args.midi)},
        "vocal": {"path": str(args.vocal), "sha256": sha(args.vocal),
                  "rate_in": int(vocal_rate), "frames_in": len(vocal),
                  "duration_seconds": len(vocal) / int(vocal_rate),
                  "mix_rate": mix_rate, "mix_frames": n},
        "target_instrumental_rms": args.target_rms,
        "raw_instrumental_rms": raw_rms,
        "instrumental_gain_db": round(float(20 * np.log10(gain)), 6) if gain else None,
        "instrumental_peak": round(float(np.max(np.abs(instrumental))), 6),
        "instrumental_rms": round(float(np.sqrt(np.mean(instrumental ** 2))), 6),
        "mix_peak_before": round(peak_before, 6),
        "mix_global_gain_db": round(float(20 * np.log10(global_gain)), 6),
        "mix_peak_after": round(float(np.max(np.abs(mix))), 6),
        "mix_rms": round(float(np.sqrt(np.mean(mix ** 2))), 6),
        "duration_seconds": round(len(mix) / mix_rate, 6),
        "tail_fade_ms": args.tail_fade_ms,
        "time_varying_processing": False,
        "vocal_processed": False,
        "files": {"instrumental": inst_path.name, "mix": mix_path.name,
                  "instrumental_sha256": sha(inst_path), "mix_sha256": sha(mix_path)},
    }
    (out / f"{name}_render_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
