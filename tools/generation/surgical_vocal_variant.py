"""Create a vocal-only noise and high-frequency artifact A/B variant.

The source-clean WAV remains untouched. Demucs is used only to isolate the
vocal; a transparent vocal cleanup runs before the surgical reducer, and the
original instrumental stem is remixed without denoising or EQ.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "vocal_repair"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "generation"))

from hf_artifact_reducer import reduce_hf_artifacts  # noqa: E402
from noise_clean_agent import clean_stem  # noqa: E402
from stem_pipeline import separate_with_demucs  # noqa: E402


def remix_surgical(
    source: Path,
    output: Path,
    work_dir: Path,
    vocal_noise_profile: str = "transparent",
    ratio_thresh: float = 3.0,
    max_reduction_db: float = -6.0,
) -> dict:
    vocals, instrumental = separate_with_demucs(source, work_dir)
    if vocal_noise_profile == "none":
        clean_vocals = vocals
        noise_report = {"profile": "none", "output": str(vocals)}
    else:
        clean_vocals = work_dir / f"vocals_noise_{vocal_noise_profile}.wav"
        noise_result = clean_stem(
            vocals,
            clean_vocals,
            role="vocals",
            profile=vocal_noise_profile,
        )
        noise_report = {
            "profile": vocal_noise_profile,
            "output": str(clean_vocals),
            "before": noise_result.before.__dict__,
            "after": noise_result.after.__dict__,
        }

    reduced = work_dir / "vocals_hf_surgical.wav"
    report = reduce_hf_artifacts(
        clean_vocals,
        reduced,
        ratio_thresh=ratio_thresh,
        max_reduction_db=max_reduction_db,
    )

    inst, inst_sr = sf.read(instrumental, always_2d=True)
    vocal, vocal_sr = sf.read(reduced, always_2d=True)
    channels = max(inst.shape[1], vocal.shape[1])
    if inst.shape[1] == 1 and channels == 2:
        inst = np.repeat(inst, 2, axis=1)
    if vocal.shape[1] == 1 and channels == 2:
        vocal = np.repeat(vocal, 2, axis=1)
    source_sr = sf.info(source).samplerate
    if inst_sr != source_sr:
        inst = resample_poly(inst, source_sr, inst_sr, axis=0)
        inst_sr = source_sr
    if vocal_sr != source_sr:
        vocal = resample_poly(vocal, source_sr, vocal_sr, axis=0)
        vocal_sr = source_sr
    if inst_sr != vocal_sr:
        raise RuntimeError(f"Stem sample-rate mismatch after resampling: instrumental={inst_sr}, vocal={vocal_sr}")
    length = min(len(inst), len(vocal))
    mix = inst[:length] + vocal[:length]
    peak = float(np.max(np.abs(mix)))
    if peak > 0.99:
        mix = mix * (0.99 / peak)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, mix, inst_sr, subtype="PCM_16")
    return {"source": str(source), "output": str(output), "frames": report.get("frames", 0),
            "channels_preserved": report.get("channels_preserved", channels),
            "sample_rate": inst_sr, "duration_s": round(length / inst_sr, 3),
            "vocal_noise": noise_report, "hf_artifact": report}


def main() -> int:
    parser = argparse.ArgumentParser(description="Make a vocal-only HF artifact A/B variant.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--vocal-noise-profile", choices=("none", "transparent", "review_fast", "balanced", "aggressive"), default="transparent")
    parser.add_argument("--ratio-thresh", type=float, default=3.0)
    parser.add_argument("--max-cut-db", type=float, default=-6.0)
    args = parser.parse_args()
    print(remix_surgical(
        args.input.resolve(),
        args.output.resolve(),
        args.work_dir.resolve(),
        vocal_noise_profile=args.vocal_noise_profile,
        ratio_thresh=args.ratio_thresh,
        max_reduction_db=args.max_cut_db,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
