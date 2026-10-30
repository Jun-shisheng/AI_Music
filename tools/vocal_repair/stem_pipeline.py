from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from vocal_repair import DEFAULT_INPUT_DIR, FILTERS, PROJECT_ROOT, _ffmpeg
from vocal_repair import analyze_file as analyze_vocal_artifacts
from rvc_convert import convert_vocals, is_rvc_configured
from noise_clean_agent import clean_stems
from humanize_agent import DEFAULT_REFS_DIR as DEFAULT_HUMAN_REFS_DIR, humanize_vocals


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vocal_repair" / "stem_pipeline"
DEFAULT_OPTIMIZED_DIR = PROJECT_ROOT / "outputs" / "vocal_repair" / "optimized_queue"


def _python() -> str:
    return sys.executable


def has_demucs() -> bool:
    try:
        subprocess.run(
            [_python(), "-c", "import demucs"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def separate_with_demucs(input_wav: Path, work_dir: Path, model: str = "htdemucs") -> tuple[Path, Path]:
    stem_dir = work_dir / "demucs" / model / input_wav.stem
    vocals = stem_dir / "vocals.wav"
    instrumental = stem_dir / "no_vocals.wav"
    if vocals.exists() and instrumental.exists():
        return vocals, instrumental

    if not has_demucs():
        raise RuntimeError(
            "demucs is not installed. Install it in the active Python environment to enable stem separation."
        )

    demucs_out = work_dir / "demucs"
    subprocess.run(
        [
            _python(),
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "-n",
            model,
            "-o",
            str(demucs_out),
            str(input_wav),
        ],
        check=True,
    )

    stem_dir = demucs_out / model / input_wav.stem
    vocals = stem_dir / "vocals.wav"
    instrumental = stem_dir / "no_vocals.wav"
    if not vocals.exists() or not instrumental.exists():
        raise FileNotFoundError(f"Demucs did not produce expected stems under {stem_dir}")
    return vocals, instrumental


def repair_vocal_stem(vocals: Path, output_path: Path, profile: str = "gentle") -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-i",
            str(vocals),
            "-af",
            FILTERS[profile],
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_path


def remix(
    instrumental: Path,
    vocals: Path,
    output_path: Path,
    vocal_gain_db: float = 0.0,
    vocal_intro_mute_seconds: float = 0.0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocal_filters = [f"volume={vocal_gain_db}dB"]
    if vocal_intro_mute_seconds > 0:
        vocal_filters.append(f"volume=enable='lt(t,{vocal_intro_mute_seconds})':volume=0")
    filter_complex = (
        f"[1:a]{','.join(vocal_filters)}[v];"
        "[0:a][v]amix=inputs=2:duration=longest:dropout_transition=0,"
        "alimiter=limit=0.95,loudnorm=I=-16:TP=-1.5:LRA=11[out]"
    )
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-i",
            str(instrumental),
            "-i",
            str(vocals),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_path


def process_file(
    input_wav: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    optimized_dir: Path = DEFAULT_OPTIMIZED_DIR,
    profile: str = "gentle",
    vocal_gain_db: float = 0.0,
    demucs_model: str = "htdemucs",
    force: bool = False,
    noise_agent: bool = True,
    noise_profile: str = "transparent",
    vocal_repair: bool = True,
    vocal_intro_mute_seconds: float = 0.0,
    vocal_artifact_max_score: int = 0,
    humanize: bool = False,
    human_refs_dir: Path = DEFAULT_HUMAN_REFS_DIR,
    rvc: bool | None = None,
    rvc_model: str | None = None,
    rvc_index: str | None = None,
    rvc_infer_cmd: str | None = None,
) -> dict:
    song_dir = output_dir / input_wav.stem
    song_dir.mkdir(parents=True, exist_ok=True)
    remix_target = song_dir / "remix.wav"
    optimized_target = optimized_dir / input_wav.name
    if remix_target.exists() and not force:
        optimized_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remix_target, optimized_target)
        return {
            "input": str(input_wav),
            "status": "skipped_existing",
            "steps": {
                "remix": {"remix": str(remix_target)},
                "optimized_queue": {"file": str(optimized_target)},
            },
        }

    metadata = {
        "input": str(input_wav),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profile": profile,
        "vocal_gain_db": vocal_gain_db,
        "vocal_intro_mute_seconds": vocal_intro_mute_seconds,
        "status": "started",
        "steps": {},
    }

    vocals, instrumental = separate_with_demucs(input_wav, song_dir, model=demucs_model)
    metadata["steps"]["separation"] = {
        "vocals": str(vocals),
        "instrumental": str(instrumental),
        "model": demucs_model,
    }

    if noise_agent:
        noise_report = clean_stems(vocals, instrumental, song_dir / "noise_agent", profile=noise_profile)
        vocals = Path(noise_report["vocals"]["output"])
        instrumental = Path(noise_report["instrumental"]["output"])
        metadata["steps"]["noise_agent"] = {
            "profile": noise_profile,
            "vocals": str(vocals),
            "instrumental": str(instrumental),
            "report": str(song_dir / "noise_agent" / "noise_clean_report.json"),
        }
    else:
        metadata["steps"]["noise_agent"] = {"status": "skipped"}

    if vocal_repair:
        repaired_vocals = repair_vocal_stem(vocals, song_dir / "repaired_vocals.wav", profile=profile)
        metadata["steps"]["vocal_repair"] = {"repaired_vocals": str(repaired_vocals)}
    else:
        repaired_vocals = vocals
        metadata["steps"]["vocal_repair"] = {"status": "skipped", "vocals": str(vocals)}

    vocals_for_remix = repaired_vocals
    if humanize:
        humanized_vocals = song_dir / "humanized_vocals.wav"
        humanize_report = humanize_vocals(repaired_vocals, humanized_vocals, refs_dir=human_refs_dir)
        vocals_for_remix = humanized_vocals
        metadata["steps"]["humanize"] = {
            "status": "succeeded",
            "vocals": str(humanized_vocals),
            "report": str(humanized_vocals.parent / "humanize_report.json"),
            "reference_count": humanize_report["profile"]["reference_count"],
            "filter_chain": humanize_report["profile"]["filter_chain"],
        }
    else:
        metadata["steps"]["humanize"] = {"status": "skipped"}

    if vocal_artifact_max_score > 0:
        vocal_artifacts = analyze_vocal_artifacts(vocals_for_remix)
        metadata["steps"]["vocal_artifact_gate"] = {
            "artifact_score": vocal_artifacts.artifact_score,
            "status": vocal_artifacts.status,
            "reasons": vocal_artifacts.reasons,
            "max_score": vocal_artifact_max_score,
        }
        if vocal_artifacts.artifact_score > vocal_artifact_max_score:
            if optimized_target.exists():
                optimized_target.unlink()
            metadata["status"] = "failed_quality_gate"
            metadata["error"] = (
                f"vocal artifact score {vocal_artifacts.artifact_score} "
                f"exceeds {vocal_artifact_max_score}"
            )
            (song_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return metadata
    else:
        metadata["steps"]["vocal_artifact_gate"] = {"status": "skipped"}

    if rvc is False:
        want_rvc = False
    elif rvc is True:
        want_rvc = True
    else:
        want_rvc = False

    if want_rvc:
        rvc_out = song_dir / "rvc_vocals.wav"
        rvc_result = convert_vocals(
            vocals_for_remix,
            rvc_out,
            enabled=True,
            model_path=rvc_model,
            index_path=rvc_index,
            infer_cmd=rvc_infer_cmd,
        )
        metadata["steps"]["rvc"] = {
            "status": rvc_result.status,
            "input": rvc_result.input,
            "output": rvc_result.output,
            "command": rvc_result.command,
            "error": rvc_result.error,
            "preset": rvc_result.preset,
        }
        if rvc_result.status == "succeeded" and rvc_result.output:
            vocals_for_remix = Path(rvc_result.output)
        elif rvc is True:
            metadata["status"] = "failed"
            metadata["error"] = rvc_result.error or rvc_result.status
            (song_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return metadata
    else:
        metadata["steps"]["rvc"] = {"status": "skipped_not_configured"}

    remix_path = remix(
        instrumental,
        vocals_for_remix,
        song_dir / "remix.wav",
        vocal_gain_db=vocal_gain_db,
        vocal_intro_mute_seconds=vocal_intro_mute_seconds,
    )
    metadata["steps"]["remix"] = {"remix": str(remix_path), "vocals_used": str(vocals_for_remix)}
    optimized_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(remix_path, optimized_target)
    metadata["steps"]["optimized_queue"] = {"file": str(optimized_target)}
    metadata["status"] = "succeeded"
    (song_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Separate vocals, repair vocal stem, and remix generated songs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--optimized-dir", type=Path, default=DEFAULT_OPTIMIZED_DIR)
    parser.add_argument("--profile", choices=sorted(FILTERS), default="transparent")
    parser.add_argument("--vocal-gain-db", type=float, default=0.0)
    parser.add_argument(
        "--vocal-intro-mute-seconds",
        type=float,
        default=0.0,
        help="Mute the separated vocal stem at the start to suppress generated hums before verse entry.",
    )
    parser.add_argument(
        "--vocal-artifact-max-score",
        type=int,
        default=0,
        help="Reject files whose repaired vocal stem artifact score exceeds this value; 0 disables the gate.",
    )
    parser.add_argument("--demucs-model", default="htdemucs")
    parser.add_argument("--limit", type=int, default=0, help="0 means all files")
    parser.add_argument("--force", action="store_true", help="Re-run separation even when remix.wav already exists.")
    parser.add_argument("--no-noise-agent", action="store_true", help="Skip stem noise cleanup before vocal repair.")
    parser.add_argument("--skip-vocal-repair", action="store_true", help="Skip the regular vocal repair filter chain.")
    parser.add_argument(
        "--noise-profile",
        choices=["transparent", "review_fast", "balanced", "aggressive"],
        default="transparent",
        help="Noise cleanup strength for separated stems.",
    )
    parser.add_argument("--rvc", action="store_true", help="Require RVC conversion (fail if not configured/succeeds).")
    parser.add_argument("--humanize", action="store_true", help="Apply human-reference vocal humanization before remix.")
    parser.add_argument("--human-refs-dir", type=Path, default=DEFAULT_HUMAN_REFS_DIR)
    parser.add_argument("--no-rvc", action="store_true", help="Disable RVC even if Cover_vision auto-detect finds a model.")
    parser.add_argument("--rvc-model", default=None, help="Path to RVC model (.pth); or set RVC_MODEL_PATH.")
    parser.add_argument("--rvc-index", default=None, help="Optional RVC index path; or set RVC_INDEX_PATH.")
    parser.add_argument(
        "--rvc-infer-cmd",
        default=None,
        help="Command template with {input} {output} {model} {index}; or set RVC_INFER_CMD.",
    )
    args = parser.parse_args()

    if args.rvc and args.no_rvc:
        parser.error("Use only one of --rvc / --no-rvc")
    rvc_mode: bool | None
    if args.no_rvc:
        rvc_mode = False
    elif args.rvc:
        rvc_mode = True
    else:
        rvc_mode = None

    wavs = sorted(args.input_dir.glob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    if not wavs:
        print(f"No wav files found in {args.input_dir}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "optimized_dir": str(args.optimized_dir),
        "demucs_available": has_demucs(),
        "items": [],
    }

    for wav in wavs:
        try:
            report["items"].append(
                process_file(
                    wav,
                    output_dir=args.output_dir,
                    optimized_dir=args.optimized_dir,
                    profile=args.profile,
                    vocal_gain_db=args.vocal_gain_db,
                    demucs_model=args.demucs_model,
                    force=args.force,
                    noise_agent=not args.no_noise_agent,
                    noise_profile=args.noise_profile,
                    vocal_repair=not args.skip_vocal_repair,
                    vocal_intro_mute_seconds=args.vocal_intro_mute_seconds,
                    vocal_artifact_max_score=args.vocal_artifact_max_score,
                    humanize=args.humanize,
                    human_refs_dir=args.human_refs_dir,
                    rvc=rvc_mode,
                    rvc_model=args.rvc_model,
                    rvc_index=args.rvc_index,
                    rvc_infer_cmd=args.rvc_infer_cmd,
                )
            )
        except Exception as exc:  # noqa: BLE001 - batch report should keep going.
            report["items"].append({"input": str(wav), "status": "failed", "error": str(exc)})

    report_path = args.output_dir / f"stem_pipeline_report_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
