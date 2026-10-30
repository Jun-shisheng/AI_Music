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
SOULX_MODEL = SOULX_ROOT / "pretrained_models" / "SoulX-Singer" / "model-svc.pt"
SOULX_CONFIG = SOULX_ROOT / "soulxsinger" / "config" / "soulxsinger.yaml"
SOULX_RMVPE = SOULX_ROOT / "pretrained_models" / "SoulX-Singer-Preprocess" / "rmvpe" / "rmvpe.pt"
SOULX_AUDIO = SOULX_ROOT / "example" / "audio"

DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "top_songs"
DEFAULT_WORK_DIR = PROJECT_ROOT / "outputs" / "soulx_auto_work"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "outputs" / "soulx_review"


@dataclass
class SoulxResult:
    input: str
    status: str
    review_mix: str | None
    generated_vocals: str | None
    artifact_score_before: int | None
    artifact_score_after: int | None
    duration_seconds: float
    mode: str
    error: str | None = None


def _safe_name(text: str, fallback: str = "song") -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", text, flags=re.ASCII).strip("._")
    return (cleaned or fallback)[:80]


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def _prompt_assets(input_wav: Path) -> tuple[Path, Path]:
    text = input_wav.stem
    lyric_file = input_wav.with_suffix(".txt")
    if lyric_file.exists():
        text += "\n" + lyric_file.read_text(encoding="utf-8", errors="ignore")
    if _has_cjk(text):
        return SOULX_AUDIO / "zh_prompt.mp3", SOULX_AUDIO / "zh_prompt_f0.npy"
    return SOULX_AUDIO / "zh_prompt.mp3", SOULX_AUDIO / "zh_prompt_f0.npy"


def _check_soulx_ready() -> None:
    missing = [
        path
        for path in [SOULX_PYTHON, SOULX_ROOT, SOULX_MODEL, SOULX_CONFIG, SOULX_RMVPE]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("SoulX-Singer is not ready: " + ", ".join(str(path) for path in missing))


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
        tail = (completed.stderr or completed.stdout or "").strip()[-4000:]
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


def _clip_audio(input_wav: Path, output_wav: Path, seconds: float | None) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [_ffmpeg(), "-y", "-i", str(input_wav)]
    if seconds and seconds > 0:
        command.extend(["-t", f"{seconds:.3f}"])
    command.extend(["-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(output_wav)])
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_wav


def _extract_f0(audio_path: Path, f0_path: Path) -> Path:
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(SOULX_ROOT)!r}); "
        "from preprocess.tools.f0_extraction import F0Extractor; "
        f"pe=F0Extractor(model_path={str(SOULX_RMVPE)!r}, device='cuda'); "
        f"pe.process({str(audio_path)!r}, f0_path={str(f0_path)!r})"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOULX_ROOT)
    _run([str(SOULX_PYTHON), "-c", script], cwd=SOULX_ROOT, env=env, timeout=900)
    if not f0_path.exists():
        raise FileNotFoundError(f"SoulX F0 extraction did not create {f0_path}")
    return f0_path


def _run_svc(target_wav: Path, target_f0: Path, save_dir: Path, n_steps: int, cfg: float) -> Path:
    prompt_wav, prompt_f0 = _prompt_assets(target_wav)
    save_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOULX_ROOT)
    _run(
        [
            str(SOULX_PYTHON),
            "-m",
            "cli.inference_svc",
            "--device",
            "cuda",
            "--model_path",
            str(SOULX_MODEL),
            "--config",
            str(SOULX_CONFIG),
            "--prompt_wav_path",
            str(prompt_wav),
            "--target_wav_path",
            str(target_wav),
            "--prompt_f0_path",
            str(prompt_f0),
            "--target_f0_path",
            str(target_f0),
            "--save_dir",
            str(save_dir),
            "--auto_shift",
            "--pitch_shift",
            "0",
            "--n_steps",
            str(n_steps),
            "--cfg",
            str(cfg),
        ],
        cwd=SOULX_ROOT,
        env=env,
        timeout=3600,
    )
    generated = save_dir / "generated.wav"
    if not generated.exists():
        raise FileNotFoundError(f"SoulX did not create {generated}")
    return generated


def _remix(instrumental: Path, generated_vocals: Path, output_wav: Path, seconds: float | None) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [_ffmpeg(), "-y", "-i", str(instrumental), "-i", str(generated_vocals)]
    if seconds and seconds > 0:
        command.extend(["-t", f"{seconds:.3f}"])
    command.extend(
        [
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=shortest:dropout_transition=0,"
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
    n_steps: int,
    cfg: float,
    force: bool,
) -> SoulxResult:
    _check_soulx_ready()
    input_wav = Path(input_wav).resolve()
    work_dir = Path(work_dir).resolve()
    review_dir = Path(review_dir).resolve()
    song_key = _safe_name(input_wav.stem, input_wav.stem[:12])
    suffix = "preview" if preview_seconds > 0 else "full"
    review_mix = review_dir / f"{song_key}_soulx_{suffix}.wav"
    generated_copy = review_dir / f"{song_key}_soulx_{suffix}_vocals.wav"
    if review_mix.exists() and generated_copy.exists() and not force:
        return SoulxResult(
            input=str(input_wav),
            status="skipped_existing",
            review_mix=str(review_mix),
            generated_vocals=str(generated_copy),
            artifact_score_before=None,
            artifact_score_after=None,
            duration_seconds=round(_ffprobe_duration(review_mix), 2),
            mode=suffix,
        )
    song_work = work_dir / song_key
    if force and song_work.exists():
        shutil.rmtree(song_work)
    song_work.mkdir(parents=True, exist_ok=True)

    try:
        vocals, instrumental = separate_with_demucs(input_wav, song_work, model="htdemucs")
        before = analyze_vocal_artifacts(vocals)

        target_vocals = _clip_audio(vocals, song_work / "target_vocals.wav", preview_seconds)
        target_instrumental = _clip_audio(instrumental, song_work / "target_instrumental.wav", preview_seconds)
        target_f0 = _extract_f0(target_vocals, song_work / "target_vocals_f0.npy")
        generated = _run_svc(target_vocals, target_f0, song_work / "svc", n_steps=n_steps, cfg=cfg)
        after = analyze_vocal_artifacts(generated)

        review_name = f"{song_key}_soulx_{suffix}.wav"
        review_mix = _remix(target_instrumental, generated, review_dir / review_name, preview_seconds)
        lyric_file = input_wav.with_suffix(".txt")
        if lyric_file.exists():
            shutil.copy2(lyric_file, review_mix.with_suffix(".txt"))
        shutil.copy2(generated, review_dir / f"{song_key}_soulx_{suffix}_vocals.wav")
        return SoulxResult(
            input=str(input_wav),
            status="succeeded",
            review_mix=str(review_mix),
            generated_vocals=str(review_dir / f"{song_key}_soulx_{suffix}_vocals.wav"),
            artifact_score_before=before.artifact_score,
            artifact_score_after=after.artifact_score,
            duration_seconds=round(_ffprobe_duration(review_mix), 2),
            mode=suffix,
        )
    except Exception as exc:  # noqa: BLE001 - workflow should report and continue.
        return SoulxResult(
            input=str(input_wav),
            status="failed",
            review_mix=None,
            generated_vocals=None,
            artifact_score_before=None,
            artifact_score_after=None,
            duration_seconds=0,
            mode="preview" if preview_seconds > 0 else "full",
            error=str(exc),
        )


def run(
    input_dir: Path,
    work_dir: Path,
    review_dir: Path,
    *,
    limit: int,
    preview_seconds: float,
    n_steps: int,
    cfg: float,
    force: bool,
    clean_review: bool,
    keep_work: bool,
) -> list[SoulxResult]:
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
        process_file(wav, work_dir, review_dir, preview_seconds=preview_seconds, n_steps=n_steps, cfg=cfg, force=force)
        for wav in wavs
    ]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "review_dir": str(review_dir),
        "preview_seconds": preview_seconds,
        "n_steps": n_steps,
        "cfg": cfg,
        "items": [asdict(result) for result in results],
    }
    (review_dir / "soulx_review_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not keep_work and work_dir.exists():
        shutil.rmtree(work_dir)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Automatic SoulX-Singer SVC review workflow for high-score songs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--preview-seconds", type=float, default=30.0, help="0 means full song.")
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-clean-review", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    results = run(
        args.input_dir,
        args.work_dir,
        args.review_dir,
        limit=args.limit,
        preview_seconds=args.preview_seconds,
        n_steps=args.n_steps,
        cfg=args.cfg,
        force=args.force,
        clean_review=not args.no_clean_review,
        keep_work=args.keep_work,
    )
    for result in results:
        print(
            f"{result.status:9s} before={result.artifact_score_before} after={result.artifact_score_after} "
            f"{Path(result.input).name}"
        )
        if result.error:
            print(result.error[-500:])
    ok_statuses = {"succeeded", "skipped_existing"}
    return 0 if all(result.status in ok_statuses for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
