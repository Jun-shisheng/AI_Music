from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from stem_pipeline import separate_with_demucs
from vocal_repair import PROJECT_ROOT, _ffmpeg, analyze_file as analyze_vocal_artifacts


SOULX_ROOT = Path(os.environ.get("SOULX_ROOT", r"G:\Cover_vision\SoulX-Singer"))
SOULX_PYTHON = Path(os.environ.get("SOULX_PYTHON", r"D:\Miniconda3\download\envs\soulxsinger\python.exe"))
SOULX_MODEL = SOULX_ROOT / "pretrained_models" / "SoulX-Singer" / "model.pt"
SOULX_CONFIG = SOULX_ROOT / "soulxsinger" / "config" / "soulxsinger.yaml"
SOULX_PHONESET = SOULX_ROOT / "soulxsinger" / "utils" / "phoneme" / "phone_set.json"
SOULX_AUDIO = SOULX_ROOT / "example" / "audio"

DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "quality25_trial"
DEFAULT_WORK_DIR = PROJECT_ROOT / "outputs" / "svs_work"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "outputs" / "svs_review"


@dataclass
class SvsResult:
    input: str
    status: str
    review_mix: str | None
    generated_vocals: str | None
    artifact_score_before: int | None
    artifact_score_after: int | None
    duration_seconds: float
    mode: str
    language: str
    vocal_polish_profile: str
    error: str | None = None


def _safe_name(text: str, fallback: str = "song") -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", text, flags=re.ASCII).strip("._")
    return (cleaned or fallback)[:80]


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def _language_for(input_wav: Path) -> str:
    text = input_wav.stem
    lyric_file = input_wav.with_suffix(".txt")
    if lyric_file.exists():
        text += "\n" + lyric_file.read_text(encoding="utf-8", errors="ignore")
    return "Mandarin" if _has_cjk(text) else "English"


def _check_soulx_ready() -> None:
    missing = [
        path
        for path in [SOULX_PYTHON, SOULX_ROOT, SOULX_MODEL, SOULX_CONFIG, SOULX_PHONESET]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("SoulX-Singer SVS is not ready: " + ", ".join(str(path) for path in missing))


def _run(command: list[str], cwd: Path, *, env: dict[str, str] | None = None, timeout: int | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()[-5000:]
        raise RuntimeError(tail or f"command failed with exit code {completed.returncode}")


def _ffprobe_duration(path: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    return float(output)


def _clip_audio(input_wav: Path, output_wav: Path, seconds: float | None, *, mono: bool = False) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [_ffmpeg(), "-y", "-i", str(input_wav)]
    if seconds and seconds > 0:
        command.extend(["-t", f"{seconds:.3f}"])
    command.extend(["-ar", "48000", "-ac", "1" if mono else "2", "-c:a", "pcm_s16le", str(output_wav)])
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_wav


def _soulx_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOULX_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_preprocess(target_vocals: Path, preprocess_dir: Path, language: str, max_merge_ms: int) -> Path:
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    target_metadata = target_vocals.with_suffix(".json")
    if target_metadata.exists():
        target_metadata.unlink()
    _run(
        [
            str(SOULX_PYTHON),
            "-m",
            "preprocess.pipeline",
            "--audio_path",
            str(target_vocals),
            "--save_dir",
            str(preprocess_dir),
            "--language",
            language,
            "--device",
            "cuda",
            "--vocal_sep",
            "False",
            "--max_merge_duration",
            str(max_merge_ms),
            "--midi_transcribe",
            "True",
        ],
        cwd=SOULX_ROOT,
        env=_soulx_env(),
        timeout=3600,
    )
    if not target_metadata.exists():
        fallback = preprocess_dir / "metadata.json"
        if fallback.exists():
            shutil.copy2(fallback, target_metadata)
    if not target_metadata.exists():
        raise FileNotFoundError(f"SoulX preprocess did not create {target_metadata}")
    meta = json.loads(target_metadata.read_text(encoding="utf-8"))
    if not meta:
        raise RuntimeError("SoulX preprocess returned empty metadata.")
    return target_metadata


def _run_svs(target_metadata: Path, save_dir: Path, language: str, use_fp16: bool) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    prompt_wav = SOULX_AUDIO / "zh_prompt.mp3"
    prompt_metadata = SOULX_AUDIO / "zh_prompt.json"
    if language == "English":
        en_prompt = SOULX_AUDIO / "en_prompt.mp3"
        en_meta = SOULX_AUDIO / "en_prompt.json"
        if en_prompt.exists() and en_meta.exists():
            prompt_wav, prompt_metadata = en_prompt, en_meta
    command = [
        str(SOULX_PYTHON),
        "-m",
        "cli.inference",
        "--device",
        "cuda",
        "--model_path",
        str(SOULX_MODEL),
        "--config",
        str(SOULX_CONFIG),
        "--prompt_wav_path",
        str(prompt_wav),
        "--prompt_metadata_path",
        str(prompt_metadata),
        "--target_metadata_path",
        str(target_metadata),
        "--phoneset_path",
        str(SOULX_PHONESET),
        "--save_dir",
        str(save_dir),
        "--auto_shift",
        "--pitch_shift",
        "0",
        "--control",
        "melody",
    ]
    if use_fp16:
        command.append("--fp16")
    _run(command, cwd=SOULX_ROOT, env=_soulx_env(), timeout=3600)
    generated = save_dir / "generated.wav"
    if not generated.exists():
        raise FileNotFoundError(f"SoulX SVS did not create {generated}")
    return generated


VOCAL_POLISH_FILTERS = {
    "none": "",
    # Lightly reduces nasal/formant buildup and brittle synthesis edges without changing the backing track.
    "natural": (
        "highpass=f=70,"
        "equalizer=f=280:t=q:w=1.2:g=-1.8,"
        "equalizer=f=430:t=q:w=1.0:g=-2.2,"
        "equalizer=f=950:t=q:w=1.4:g=-1.2,"
        "equalizer=f=3600:t=q:w=1.0:g=-1.0,"
        "alimiter=limit=0.92"
    ),
    "clear_natural": (
        "highpass=f=70,"
        "equalizer=f=280:t=q:w=1.2:g=-1.6,"
        "equalizer=f=430:t=q:w=1.0:g=-2.0,"
        "equalizer=f=950:t=q:w=1.3:g=-0.8,"
        "equalizer=f=2400:t=q:w=1.0:g=0.9,"
        "equalizer=f=5200:t=q:w=1.0:g=-0.5,"
        "alimiter=limit=0.92"
    ),
    "soft": (
        "highpass=f=75,"
        "equalizer=f=260:t=q:w=1.0:g=-2.5,"
        "equalizer=f=420:t=q:w=0.9:g=-3.0,"
        "equalizer=f=1050:t=q:w=1.2:g=-1.5,"
        "equalizer=f=4200:t=q:w=1.0:g=-1.8,"
        "alimiter=limit=0.90"
    ),
}


def _polish_vocals(generated_vocals: Path, output_wav: Path, profile: str) -> Path:
    if profile == "none":
        return generated_vocals
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-i",
            str(generated_vocals),
            "-af",
            VOCAL_POLISH_FILTERS[profile],
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_wav


def _remix(instrumental: Path, generated_vocals: Path, output_wav: Path, seconds: float | None) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [_ffmpeg(), "-y", "-i", str(instrumental), "-i", str(generated_vocals)]
    if seconds and seconds > 0:
        command.extend(["-t", f"{seconds:.3f}"])
    command.extend(
        [
            "-filter_complex",
            "[1:a]volume=-1.5dB[v];"
            "[0:a][v]amix=inputs=2:duration=shortest:dropout_transition=0,"
            "alimiter=limit=0.95,loudnorm=I=-16:TP=-1.5:LRA=11[out]",
            "-map",
            "[out]",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
    )
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_wav


def process_file(
    input_wav: Path,
    work_dir: Path,
    review_dir: Path,
    *,
    preview_seconds: float,
    force: bool,
    use_fp16: bool,
    vocal_polish_profile: str,
) -> SvsResult:
    _check_soulx_ready()
    input_wav = Path(input_wav).resolve()
    work_dir = Path(work_dir).resolve()
    review_dir = Path(review_dir).resolve()
    song_key = _safe_name(input_wav.stem, input_wav.stem[:12])
    suffix = "preview" if preview_seconds > 0 else "full"
    profile_suffix = "" if vocal_polish_profile == "none" else f"_{vocal_polish_profile}"
    language = _language_for(input_wav)
    review_mix = review_dir / f"{song_key}_soulx_svs_{suffix}{profile_suffix}.wav"
    generated_copy = review_dir / f"{song_key}_soulx_svs_{suffix}{profile_suffix}_vocals.wav"
    if review_mix.exists() and generated_copy.exists() and not force:
        return SvsResult(
            input=str(input_wav),
            status="skipped_existing",
            review_mix=str(review_mix),
            generated_vocals=str(generated_copy),
            artifact_score_before=None,
            artifact_score_after=None,
            duration_seconds=round(_ffprobe_duration(review_mix), 2),
            mode=suffix,
            language=language,
            vocal_polish_profile=vocal_polish_profile,
        )

    song_work = work_dir / song_key
    if force and song_work.exists():
        shutil.rmtree(song_work)
    song_work.mkdir(parents=True, exist_ok=True)

    try:
        vocals, instrumental = separate_with_demucs(input_wav, song_work, model="htdemucs")
        before = analyze_vocal_artifacts(vocals)
        target_vocals = _clip_audio(vocals, song_work / "target_vocals.wav", preview_seconds, mono=True)
        target_instrumental = _clip_audio(instrumental, song_work / "target_instrumental.wav", preview_seconds)
        target_metadata = _run_preprocess(
            target_vocals,
            song_work / "preprocess",
            language=language,
            max_merge_ms=max(1000, int((preview_seconds or 60) * 1000)),
        )
        generated = _run_svs(target_metadata, song_work / "svs", language=language, use_fp16=use_fp16)
        polished = _polish_vocals(generated, song_work / f"generated_{vocal_polish_profile}.wav", vocal_polish_profile)
        after = analyze_vocal_artifacts(polished)

        review_mix = _remix(target_instrumental, polished, review_mix, preview_seconds)
        lyric_file = input_wav.with_suffix(".txt")
        if lyric_file.exists():
            shutil.copy2(lyric_file, review_mix.with_suffix(".txt"))
        shutil.copy2(polished, generated_copy)
        return SvsResult(
            input=str(input_wav),
            status="succeeded",
            review_mix=str(review_mix),
            generated_vocals=str(generated_copy),
            artifact_score_before=before.artifact_score,
            artifact_score_after=after.artifact_score,
            duration_seconds=round(_ffprobe_duration(review_mix), 2),
            mode=suffix,
            language=language,
            vocal_polish_profile=vocal_polish_profile,
        )
    except Exception as exc:  # noqa: BLE001 - batch workflow should report and continue.
        return SvsResult(
            input=str(input_wav),
            status="failed",
            review_mix=None,
            generated_vocals=None,
            artifact_score_before=None,
            artifact_score_after=None,
            duration_seconds=0,
            mode=suffix,
            language=language,
            vocal_polish_profile=vocal_polish_profile,
            error=str(exc),
        )


def run(
    input_dir: Path,
    work_dir: Path,
    review_dir: Path,
    *,
    limit: int,
    preview_seconds: float,
    force: bool,
    clean_review: bool,
    keep_work: bool,
    use_fp16: bool,
    vocal_polish_profile: str,
) -> list[SvsResult]:
    input_dir = Path(input_dir).resolve()
    work_dir = Path(work_dir).resolve()
    review_dir = Path(review_dir).resolve()
    if clean_review and review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(input_dir.glob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    results = [
        process_file(
            wav,
            work_dir,
            review_dir,
            preview_seconds=preview_seconds,
            force=force,
            use_fp16=use_fp16,
            vocal_polish_profile=vocal_polish_profile,
        )
        for wav in wavs
    ]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "review_dir": str(review_dir),
        "preview_seconds": preview_seconds,
        "vocal_polish_profile": vocal_polish_profile,
        "items": [asdict(result) for result in results],
    }
    (review_dir / "svs_review_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not keep_work and work_dir.exists():
        shutil.rmtree(work_dir)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="SoulX-Singer SVS re-singing review workflow for high-score songs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--preview-seconds", type=float, default=30.0, help="0 means full song.")
    parser.add_argument("--vocal-polish-profile", choices=sorted(VOCAL_POLISH_FILTERS), default="none")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-clean-review", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    results = run(
        args.input_dir,
        args.work_dir,
        args.review_dir,
        limit=args.limit,
        preview_seconds=args.preview_seconds,
        force=args.force,
        clean_review=not args.no_clean_review,
        keep_work=args.keep_work,
        use_fp16=args.fp16,
        vocal_polish_profile=args.vocal_polish_profile,
    )
    for result in results:
        print(
            f"{result.status:9s} before={result.artifact_score_before} after={result.artifact_score_after} "
            f"lang={result.language} {Path(result.input).name}"
        )
        if result.error:
            print(result.error[-800:])
    ok_statuses = {"succeeded", "skipped_existing"}
    return 0 if all(result.status in ok_statuses for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
