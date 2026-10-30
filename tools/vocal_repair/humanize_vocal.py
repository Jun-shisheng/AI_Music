"""Render humanized vocal variants of a song for A/B listening.

Each variant applies a different vocal treatment chain to the demucs-separated
vocal stem, then remixes with the instrumental:

- baseline: frame repair only (0.22 attenuation), no EQ
- air: frame repair (0.5) + natural formant EQ + 7k air shelf
- air_sat: air + gentle tanh soft saturation (analog warmth)
- de_buzz: air + deesser to tame loud sibilance/buzz

Usage:
    python humanize_vocal.py <song.wav> <output_dir> [--work-dir ...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "vocal_repair"))

from frame_repair import repair_vocals  # noqa: E402
from high_note_tamer import tame_high_notes  # noqa: E402
from noise_floor_reducer import reduce_noise_floor  # noqa: E402
from stem_pipeline import remix, separate_with_demucs  # noqa: E402

NATURAL_AIR_EQ = (
    "highpass=f=70,"
    "equalizer=f=280:t=q:w=1.2:g=-1.8,"
    "equalizer=f=430:t=q:w=1.0:g=-2.2,"
    "equalizer=f=950:t=q:w=1.4:g=-1.2,"
    "equalizer=f=3600:t=q:w=1.0:g=-1.0,"
    "highshelf=f=7000:g=2.5"
)
DE_BUZZ_TAIL = "deesser=i=0.15:m=0.30:f=0.45,alimiter=limit=0.94"

# Preferred humanize chain (selected by user listening A/B): natural formant
# EQ + air shelf + deesser for high-frequency sibilance/buzz.
HUMANIZE_VARIANTS: dict[str, str] = {
    "de_buzz": NATURAL_AIR_EQ + "," + DE_BUZZ_TAIL,
}


def _tanh_saturation(wav_in: Path, wav_out: Path, drive: float = 1.4) -> None:
    y, sr = sf.read(wav_in, always_2d=True)
    x = y.astype(np.float64) / max(np.max(np.abs(y)), 1e-9)
    sat = np.tanh(drive * x) / np.tanh(drive)
    # blend 75% saturated with 25% dry to keep it subtle
    out = (0.75 * sat + 0.25 * x)
    sf.write(wav_out, out, sr, subtype="PCM_16")


def _ffmpeg_af(src: Path, dst: Path, filter_chain: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-af", filter_chain, "-ar", "48000", "-ac", "2", str(dst)],
        check=True,
    )


def humanize_stem(vocals: Path, output: Path, variant: str = "de_buzz") -> Path:
    """Reduce noise floor, EQ/tame, then frame-repair last (isolation mode).

    Order matters: the de_buzz chain's 7 kHz air shelf re-introduces harsh
    high-frequency noise when applied AFTER frame repair. Frame repair must be
    the final vocal step so defect regions are suppressed after the EQ stage.

    Isolation mode (default): the zcr harsh detector cannot separate sibilants
    from machine noise — human vocal references trigger 14-149 'harsh' events
    per file because s/sh/ch naturally have high zcr. Repairing only isolated
    single-frame bursts halves the 5-8k diction loss (iso -1.0 dB vs standard
    -1.6/-1.8 dB) while still removing true burst noise. Use variant
    "repair_std2" for the older aggressive double-pass cleanup.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    reduced = output.with_name(f"{output.stem}_noise_reduced.wav")
    reduce_noise_floor(vocals, reduced)
    eqd = output.with_name(f"{output.stem}_eq.wav")
    chain = HUMANIZE_VARIANTS[variant] + ",alimiter=limit=0.94"
    _ffmpeg_af(reduced, eqd, chain)
    tamed = output.with_name(f"{output.stem}_tamed.wav")
    tame_high_notes(eqd, tamed)
    if variant == "repair_std2":
        repair_vocals(tamed, output, repair_rounds=2)
    else:
        repair_vocals(tamed, output, repair_rounds=1, isolation_only=True)
    reduced.unlink(missing_ok=True)
    eqd.unlink(missing_ok=True)
    tamed.unlink(missing_ok=True)
    return output


def render_variants(song: Path, out_dir: Path, work_dir: Path, keep_work: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    song_work = work_dir / song.stem
    song_work.mkdir(parents=True, exist_ok=True)
    vocals, instrumental = separate_with_demucs(song, song_work)

    variants: dict[str, str | None] = {
        "baseline": None,       # frame repair only
        "air": NATURAL_AIR_EQ,
        "air_sat": None,        # saturation + air, applied in python+ffmpeg
        "de_buzz": HUMANIZE_VARIANTS["de_buzz"],
    }

    repaired = song_work / "vocals_frame_repaired.wav"
    repair_vocals(vocals, repaired)

    results = {}
    for name, eq in variants.items():
        staged = song_work / f"vocals_{name}.wav"
        if name == "de_buzz":
            # match the production chain: noise floor -> EQ -> tame -> repair last
            humanize_stem(vocals, staged, variant="de_buzz")
        elif name == "baseline":
            staged = repaired
        elif name == "air":
            _ffmpeg_af(repaired, staged, eq + ",alimiter=limit=0.94")
        elif name == "air_sat":
            sat = song_work / "vocals_air_sat_stage.wav"
            _ffmpeg_af(repaired, sat, NATURAL_AIR_EQ + ",alimiter=limit=0.94")
            _tanh_saturation(sat, staged)
        output = out_dir / f"{song.stem}_{name}.wav"
        remix(instrumental, staged, output)
        results[name] = str(output)

    if not keep_work and work_dir.exists():
        import shutil
        shutil.rmtree(work_dir)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("song", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "humanize_listen")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "outputs" / "humanize_work")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--variant", choices=[*HUMANIZE_VARIANTS, "all"], default="all")
    parser.add_argument("--output", type=Path, default=None,
                        help="Exact output path for single-variant mode (overrides --output-dir naming).")
    args = parser.parse_args()
    if args.variant == "all":
        results = render_variants(args.song, args.output_dir, args.work_dir, args.keep_work)
        for name, path in results.items():
            print(f"{name:10s} -> {path}")
    else:
        # single variant: separate, humanize, remix to output-dir/<stem>_<variant>.wav or --output
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        song_work = args.work_dir / args.song.stem
        song_work.mkdir(parents=True, exist_ok=True)
        vocals, instrumental = separate_with_demucs(args.song, song_work)
        staged = song_work / f"vocals_{args.variant}.wav"
        humanize_stem(vocals, staged, variant=args.variant)
        output = args.output or out_dir / f"{args.song.stem}_{args.variant}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        remix(instrumental, staged, output)
        if not args.keep_work and args.work_dir.exists():
            import shutil
            shutil.rmtree(args.work_dir)
        print(f"{args.variant:10s} -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
