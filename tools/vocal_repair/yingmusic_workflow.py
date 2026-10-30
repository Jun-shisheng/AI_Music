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


YING_ROOT = Path(os.environ.get("YINGMUSIC_ROOT", r"G:\Cover_vision\YingMusic-Singer"))
YING_PYTHON = Path(os.environ.get("YINGMUSIC_PYTHON", r"D:\Miniconda3\download\envs\yingsinger\python.exe"))
YING_CKPTS = YING_ROOT / "ckpts"
DEFAULT_TIMBRE_AUDIO = YING_ROOT / "resources" / "audios" / "female.wav"
DEFAULT_TIMBRE_CONTENT = "冰刀划的圈，圈起了谁改变。"

DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "quality25_trial"
DEFAULT_WORK_DIR = PROJECT_ROOT / "outputs" / "yingmusic_work"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "outputs" / "yingmusic_review"


@dataclass
class YingResult:
    input: str
    status: str
    review_mix: str | None
    generated_vocals: str | None
    artifact_score_before: int | None
    artifact_score_after: int | None
    start_seconds: float
    duration_seconds: float
    lyrics: str
    error: str | None = None


def _safe_name(text: str, fallback: str = "song") -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", text, flags=re.ASCII).strip("._")
    return (cleaned or fallback)[:80]


def _fix_mojibake(text: str) -> str:
    try:
        repaired = text.encode("gbk", errors="strict").decode("utf-8", errors="strict")
    except UnicodeError:
        return text
    original_cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    repaired_cjk = len(re.findall(r"[\u3400-\u9fff]", repaired))
    return repaired if repaired_cjk >= original_cjk else text


def _read_lyrics(path: Path, max_chars: int = 120) -> str:
    if not path.exists():
        return ""
    text = _fix_mojibake(path.read_text(encoding="utf-8", errors="ignore"))
    lines: list[str] = []
    skip_sections = {"intro", "interlude", "outro", "instrumental intro", "前奏", "间奏", "尾奏"}
    skip_current = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith(("title:", "score:", "category:")):
            continue
        match = re.fullmatch(r"\[(.+)]", line)
        if match:
            section = match.group(1).strip().lower()
            skip_current = section in skip_sections
            continue
        if skip_current:
            continue
        if re.fullmatch(r"[啊呀呜喔哦嗯哼haohum\s~,.，。！？!?-]+", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
        if sum(len(item) for item in lines) >= max_chars:
            break
    lyric = "，".join(lines)
    return lyric[:max_chars].strip("，, ")


def _check_ready() -> None:
    missing = [
        path
        for path in [
            YING_ROOT,
            YING_PYTHON,
            YING_CKPTS / "singer.v1.pt",
            YING_CKPTS / "autoencoder_music_dsp1920.ckpt",
            YING_CKPTS / "rmvpe.pt",
            YING_CKPTS / "some.pt",
            YING_CKPTS / "stable_audio_1920_vae.json",
            DEFAULT_TIMBRE_AUDIO,
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("YingMusic-Singer is not ready: " + ", ".join(str(path) for path in missing))


def _run(command: list[str], cwd: Path, *, env: dict[str, str] | None = None, timeout: int | None = None) -> None:
    completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
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


def _clip_audio(input_wav: Path, output_wav: Path, start_seconds: float, duration_seconds: float) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [_ffmpeg(), "-y"]
    if start_seconds > 0:
        command.extend(["-ss", f"{start_seconds:.3f}"])
    command.extend(["-i", str(input_wav), "-t", f"{duration_seconds:.3f}", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(output_wav)])
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_wav


def _run_yingmusic(
    melody_audio: Path,
    output_vocals: Path,
    lyrics: str,
    *,
    cfg_strength: float,
    nfe_steps: int,
    seed: int,
    pitch_shift: int,
) -> Path:
    output_vocals.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(YING_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    request_path = output_vocals.with_suffix(".request.json")
    request_path.write_text(
        json.dumps(
            {
                "ckpt_path": str(YING_CKPTS),
                "timbre_audio_path": str(DEFAULT_TIMBRE_AUDIO),
                "timbre_audio_content": DEFAULT_TIMBRE_CONTENT,
                "melody_audio_path": str(melody_audio),
                "lyrics": lyrics,
                "out_path": str(output_vocals),
                "cfg_strength": cfg_strength,
                "nfe_steps": nfe_steps,
                "seed": seed,
                "pitch_shift": pitch_shift,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    script = r"""
import json
import sys

import librosa
import soundfile as sf
import torch

sys.path.insert(0, r"G:\Cover_vision\YingMusic-Singer\src")
import singer.model as model


def load_audio_soundfile(path, target_sr):
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = audio.T
    if audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, axis=-1)
    tensor = torch.from_numpy(audio)
    if tensor.shape[0] == 1 and target_sr == model.SAMPLE_RATE_48K:
        tensor = tensor.repeat(2, 1)
    return tensor, target_sr


model.load_audio = load_audio_soundfile

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    cfg = json.load(fh)

singer = model.YingSinger(singer_path=cfg["ckpt_path"], device="cuda" if torch.cuda.is_available() else "cpu")
generated = singer.inference(
    timbre_audio_path=cfg["timbre_audio_path"],
    timbre_audio_content=cfg["timbre_audio_content"],
    melody_audio_path=cfg["melody_audio_path"],
    lyrics=cfg["lyrics"],
    pitch_shift=cfg["pitch_shift"],
    cfg_strength=cfg["cfg_strength"],
    nfe_steps=cfg["nfe_steps"],
    seed=cfg["seed"],
)
audio = generated.detach().cpu().numpy()
if audio.ndim == 2:
    audio = audio.T
sf.write(cfg["out_path"], audio, model.SAMPLE_RATE_48K)
print("Saved output to:", cfg["out_path"])
"""
    _run(
        [
            str(YING_PYTHON),
            "-c",
            script,
            str(request_path),
        ],
        cwd=YING_ROOT,
        env=env,
        timeout=3600,
    )
    if not output_vocals.exists():
        raise FileNotFoundError(f"YingMusic did not create {output_vocals}")
    return output_vocals


def _remix(instrumental: Path, vocals: Path, output_wav: Path) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-i",
            str(instrumental),
            "-i",
            str(vocals),
            "-filter_complex",
            "[1:a]highpass=f=70,volume=-1.5dB[v];"
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
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_wav


def process_file(
    input_wav: Path,
    work_dir: Path,
    review_dir: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    cfg_strength: float,
    nfe_steps: int,
    seed: int,
    pitch_shift: int,
    force: bool,
) -> YingResult:
    _check_ready()
    input_wav = input_wav.resolve()
    song_key = _safe_name(input_wav.stem, input_wav.stem[:12])
    suffix = f"{int(start_seconds)}s_{int(duration_seconds)}s"
    review_mix = review_dir / f"{song_key}_yingmusic_{suffix}.wav"
    generated_copy = review_dir / f"{song_key}_yingmusic_{suffix}_vocals.wav"
    lyrics = _read_lyrics(input_wav.with_suffix(".txt"))
    if not lyrics:
        raise RuntimeError(f"No usable lyrics found for {input_wav}")
    if review_mix.exists() and generated_copy.exists() and not force:
        return YingResult(
            input=str(input_wav),
            status="skipped_existing",
            review_mix=str(review_mix),
            generated_vocals=str(generated_copy),
            artifact_score_before=None,
            artifact_score_after=None,
            start_seconds=start_seconds,
            duration_seconds=round(_ffprobe_duration(review_mix), 2),
            lyrics=lyrics,
        )

    song_work = work_dir / song_key
    if force and song_work.exists():
        shutil.rmtree(song_work)
    song_work.mkdir(parents=True, exist_ok=True)
    try:
        vocals, instrumental = separate_with_demucs(input_wav, song_work, model="htdemucs")
        before = analyze_vocal_artifacts(vocals)
        melody_audio = _clip_audio(vocals, song_work / "melody_vocals.wav", start_seconds, duration_seconds)
        instrumental_clip = _clip_audio(instrumental, song_work / "instrumental.wav", start_seconds, duration_seconds)
        generated = _run_yingmusic(
            melody_audio,
            song_work / "yingmusic_vocals.wav",
            lyrics,
            cfg_strength=cfg_strength,
            nfe_steps=nfe_steps,
            seed=seed,
            pitch_shift=pitch_shift,
        )
        after = analyze_vocal_artifacts(generated)
        _remix(instrumental_clip, generated, review_mix)
        shutil.copy2(generated, generated_copy)
        lyric_txt = review_mix.with_suffix(".txt")
        lyric_txt.write_text(lyrics + "\n", encoding="utf-8")
        return YingResult(
            input=str(input_wav),
            status="succeeded",
            review_mix=str(review_mix),
            generated_vocals=str(generated_copy),
            artifact_score_before=before.artifact_score,
            artifact_score_after=after.artifact_score,
            start_seconds=start_seconds,
            duration_seconds=round(_ffprobe_duration(review_mix), 2),
            lyrics=lyrics,
        )
    except Exception as exc:  # noqa: BLE001 - review workflows should report failures.
        return YingResult(
            input=str(input_wav),
            status="failed",
            review_mix=None,
            generated_vocals=None,
            artifact_score_before=None,
            artifact_score_after=None,
            start_seconds=start_seconds,
            duration_seconds=0,
            lyrics=lyrics,
            error=str(exc),
        )


def run(
    input_dir: Path,
    work_dir: Path,
    review_dir: Path,
    *,
    limit: int,
    start_seconds: float,
    duration_seconds: float,
    cfg_strength: float,
    nfe_steps: int,
    seed: int,
    pitch_shift: int,
    force: bool,
    clean_review: bool,
    keep_work: bool,
) -> list[YingResult]:
    input_dir = input_dir.resolve()
    work_dir = work_dir.resolve()
    review_dir = review_dir.resolve()
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
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            cfg_strength=cfg_strength,
            nfe_steps=nfe_steps,
            seed=seed,
            pitch_shift=pitch_shift,
            force=force,
        )
        for wav in wavs
    ]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "review_dir": str(review_dir),
        "start_seconds": start_seconds,
        "duration_seconds": duration_seconds,
        "cfg_strength": cfg_strength,
        "nfe_steps": nfe_steps,
        "seed": seed,
        "pitch_shift": pitch_shift,
        "items": [asdict(result) for result in results],
    }
    (review_dir / "yingmusic_review_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not keep_work and work_dir.exists():
        shutil.rmtree(work_dir)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="YingMusic-Singer melody-guided re-singing review workflow.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--start-seconds", type=float, default=30.0)
    parser.add_argument("--duration-seconds", type=float, default=30.0, help="YingMusic recommends <=45 seconds.")
    parser.add_argument("--cfg-strength", type=float, default=4.0)
    parser.add_argument("--nfe-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--pitch-shift", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-clean-review", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    results = run(
        args.input_dir,
        args.work_dir,
        args.review_dir,
        limit=args.limit,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
        cfg_strength=args.cfg_strength,
        nfe_steps=args.nfe_steps,
        seed=args.seed,
        pitch_shift=args.pitch_shift,
        force=args.force,
        clean_review=not args.no_clean_review,
        keep_work=args.keep_work,
    )
    for result in results:
        print(
            f"{result.status:9s} before={result.artifact_score_before} after={result.artifact_score_after} "
            f"start={result.start_seconds}s dur={result.duration_seconds}s {Path(result.input).name}"
        )
        if result.error:
            print(result.error[-800:])
    ok_statuses = {"succeeded", "skipped_existing"}
    return 0 if all(result.status in ok_statuses for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
