"""Reproducible local CPU lyric QA; ASR is a proxy, never a singer/quality score.

Usage (any Python 3.10+; dependencies stay in the existing separate environments):
  python tools/music_qa.py --audio take.wav --lyrics lyrics.txt --out-dir outputs/qa/take

No target words, initial_prompt, prefix or hotwords are supplied to Whisper.
The command exits nonzero on execution/invalid-audio failures, not on subjective
quality. Read qa.json's content flags; a completed run does not approve a song.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
VERSION = 1
THREADS = 2
DEFAULT_MODEL = ROOT / "PR_AIMusic_Extension/models/faster-whisper-small"
DEFAULT_ASR = ROOT / "tools/asr/.venv/Scripts/python.exe"
DEFAULT_AUDIO = ROOT / "tools/vocal_repair/.venv/Scripts/python.exe"
FILES = ("request.json", "qa.json", "QA_SUMMARY.md", "target_lyrics.txt",
         "lyrics_input.txt", "audio.json", "transcript.json", "audio_request.json",
         "asr_request.json", "audio.log", "asr.log", "qa_manifest.json", "qa.lock")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    # English/Unicode words and individual CJK characters. No punctuation/case penalty.
    return re.findall(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+(?:'[^\W_\u3400-\u9fff]+)*", text)


def lyric_lines(text: str) -> list[str]:
    lines = []
    for raw in text.lstrip("\ufeff").splitlines():
        line = raw.strip()
        section = re.match(r"^\[[^\]]+]\s*(.*)$", line)
        if section:
            line = section[1].strip()
        if tokens(line):
            lines.append(line)
    if not lines:
        raise ValueError("Lyrics contain no words after removing section headings")
    return lines


def align_words(expected: list[str], observed: list[str]) -> list[dict]:
    """Levenshtein alignment with deterministic ties; never reuses ASR words."""
    # Byte backpointers avoid a Python-int quadratic matrix for long songs.
    trace = [bytearray(len(observed) + 1) for _ in range(len(expected) + 1)]
    previous = list(range(len(observed) + 1))
    for j in range(1, len(observed) + 1):
        trace[0][j] = 3
    for i, word in enumerate(expected, 1):
        current = [i] + [0] * len(observed)
        trace[i][0] = 2
        for j, other in enumerate(observed, 1):
            candidates = (previous[j - 1] + (word != other), previous[j] + 1, current[j - 1] + 1)
            choice = min(range(3), key=lambda k: candidates[k])
            current[j], trace[i][j] = candidates[choice], choice + 1
        previous = current
    i, j, result = len(expected), len(observed), []
    while i or j:
        choice = trace[i][j]
        if choice == 1:
            i -= 1
            j -= 1
            kind = "match" if expected[i] == observed[j] else "substitution"
            result.append(dict(operation=kind, expected_index=i, asr_index=j,
                               expected=expected[i], asr=observed[j]))
        elif choice == 2:
            i -= 1
            result.append(dict(operation="deletion", expected_index=i, asr_index=j,
                               expected=expected[i], asr=None))
        else:
            j -= 1
            result.append(dict(operation="insertion", expected_index=i, asr_index=j,
                               expected=None, asr=observed[j]))
    return result[::-1]


def compare_text(lines: list[str], transcript: str, recovery_threshold=.6) -> dict:
    target = [word for line in lines for word in tokens(line)]
    observed = tokens(transcript)
    if not target:
        raise ValueError("Target lyrics contain no words")
    alignment = align_words(target, observed)
    counts = {kind: sum(x["operation"] == kind for x in alignment)
              for kind in ("substitution", "deletion", "insertion")}
    distance = sum(counts.values())
    per_line, offset = [], 0
    for number, line in enumerate(lines, 1):
        size = len(tokens(line))
        end = offset + size
        parts = [x for x in alignment if offset <= x["expected_index"] < end or
                 (number == len(lines) and x["operation"] == "insertion" and x["expected_index"] == end)]
        matched = sum(x["operation"] == "match" for x in parts)
        changes = [x for x in parts if x["operation"] != "match"]
        per_line.append(dict(line=number, target=line, expected_words=size,
                             matched_words=matched, word_match_fraction=matched / size,
                             approximate_recovered=matched / size >= recovery_threshold,
                             asr_text_exact_match=not changes, changes=changes))
        offset = end
    exact = distance == 0
    return dict(normalized_target_words=target, normalized_asr_words=observed,
                target_word_count=len(target), asr_word_count=len(observed),
                word_edits=counts, word_edit_distance=distance,
                asr_word_edit_rate_proxy=distance / len(target),
                word_coverage=len(observed) / len(target),
                character_coverage=len("".join(observed)) / len("".join(target)),
                normalized_word_similarity=difflib.SequenceMatcher(None, target, observed, autojunk=False).ratio(),
                normalized_character_similarity=difflib.SequenceMatcher(None, "".join(target), "".join(observed), autojunk=False).ratio(),
                asr_normalized_text_exact_match=exact,
                content_status="asr_text_matches_target_unverified_audio" if exact else "asr_differences_require_review",
                approximate_lines_recovered=sum(x["approximate_recovered"] for x in per_line),
                lines_total=len(lines), line_recovery_threshold=recovery_threshold,
                line_recovery_method="matched target words in one global Levenshtein alignment; ASR words are never reused across lines",
                lines=per_line, alignment=alignment,
                note="S/D/I and recovery are ASR proxies, not verified singing errors, quality or singer-identity scores.")


def native_audio(path: Path) -> dict:
    """Stream original samples; peak is across all native channels, never a downmix."""
    import numpy as np
    import soundfile as sf

    count = bad = near = full = 0
    squares, peak = 0., 0.
    with sf.SoundFile(path) as stream:
        frames, rate, channels, subtype = len(stream), stream.samplerate, stream.channels, stream.subtype
        for data in stream.blocks(blocksize=65536, dtype="float64", always_2d=True):
            count += data.size
            finite = np.isfinite(data)
            bad += int((~finite).sum())
            valid = data[finite]
            if valid.size:
                absolute = np.abs(valid)
                peak = max(peak, float(absolute.max()))
                squares += float(np.square(valid).sum())
                near += int((absolute >= .999).sum())
                full += int((absolute >= 1.).sum())
    rms = math.sqrt(squares / count) if count and not bad else None
    errors = []
    if not frames or not count:
        errors.append("Audio has zero frames")
    if bad:
        errors.append("Audio contains nonfinite samples")
    if count and not bad and peak <= 1e-8:
        errors.append("Audio is silent; do not ask ASR to invent a transcript")
    return dict(valid=not errors, errors=errors, frames=frames, sample_rate=rate, channels=channels,
                subtype=subtype, duration_seconds=frames / rate, nonfinite_samples=bad,
                native_sample_peak=peak, rms=rms,
                rms_dbfs=20 * math.log10(max(rms, 1e-12)) if rms is not None else None,
                samples_at_or_above_0_999=near, near_full_scale_sample_ratio=near / count if count else 0,
                samples_at_or_above_1_0=full,
                peak_measurement="original native samples across all channels; no resampling or downmix",
                clipping_caveat="Full-scale samples are a review cue, not proof of audible distortion.")


def transcribe(request: dict) -> dict:
    import importlib.metadata
    import faster_whisper

    model = faster_whisper.WhisperModel(request["model"], device="cpu", compute_type="int8",
                                       cpu_threads=THREADS, num_workers=1, local_files_only=True)
    # The request deliberately has no lyrics; never bias ASR toward the answer.
    segments, info = model.transcribe(request["audio"], language=request["language"], beam_size=5,
                                      vad_filter=False, word_timestamps=True, condition_on_previous_text=False)
    rows = [dict(start=s.start, end=s.end, text=s.text, avg_logprob=s.avg_logprob,
                 no_speech_prob=s.no_speech_prob,
                 words=[dict(start=w.start, end=w.end, word=w.word, probability=w.probability)
                        for w in (s.words or [])]) for s in segments]
    return dict(text=" ".join(x["text"].strip() for x in rows).strip(), segments=rows,
                language=info.language, language_probability=info.language_probability,
                faster_whisper_version=importlib.metadata.version("faster-whisper"),
                ctranslate2_version=importlib.metadata.version("ctranslate2"),
                target_conditioning=False, initial_prompt=None, hotwords=None)


def cpu_env():
    return dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                PYTHONIOENCODING="utf-8", PYTHONUTF8="1")


def runtime_info(python: Path, packages: list[str]) -> dict:
    code = ("import sys,json,importlib.metadata as m; "
            "print(json.dumps({'python':sys.executable,'python_version':sys.version,"
            "'packages':{x:m.version(x) for x in sys.argv[1:]}}))")
    result = subprocess.run([str(python), "-I", "-c", code, *packages], env=cpu_env(),
                            check=True, capture_output=True, text=True, encoding="utf-8", timeout=30)
    return json.loads(result.stdout)


def run_worker(python: Path, kind: str, request: dict, directory: Path) -> dict:
    spec = directory / f"{kind}_request.json"
    output = directory / ("audio.json" if kind == "audio" else "transcript.json")
    write_json(spec, request)
    command = [str(python), "-I", "-u", str(Path(__file__).resolve()), "--_worker", kind,
               "--request-file", str(spec), "--output-file", str(output)]
    with (directory / f"{kind}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, env=cpu_env(), stdout=log, stderr=subprocess.STDOUT,
                                timeout=1800)
    if result.returncode:
        raise RuntimeError(f"{kind} worker failed ({result.returncode}); see {directory / (kind + '.log')}")
    return read_json(output)


@contextmanager
def output_lock(path: Path):
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def make_request(args) -> dict:
    for path in (args.audio, args.lyrics, args.asr_python, args.audio_python):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.audio in {args.out_dir / n for n in FILES} or args.lyrics in {args.out_dir / n for n in FILES}:
        raise ValueError("Input would be overwritten by a QA artifact; choose a different out-dir")
    data = args.lyrics.read_bytes()
    text = data.decode("utf-8-sig")
    lines = lyric_lines(text)
    if not (args.model / "model.bin").is_file() or not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"Local faster-whisper model is incomplete: {args.model}")
    names = ("model.bin", "config.json", "tokenizer.json", "vocabulary.json", "vocabulary.txt", "preprocessor_config.json")
    return dict(schema_version=VERSION, audio=dict(path=str(args.audio), sha256=sha(args.audio)),
                lyrics=dict(path=str(args.lyrics), sha256=hashlib.sha256(data).hexdigest(),
                            original_text=text, target_lines=lines),
                model=dict(path=str(args.model), files={name: sha(args.model / name) for name in names if (args.model / name).is_file()}),
                runtimes=dict(asr=runtime_info(args.asr_python, ["faster-whisper", "ctranslate2"]),
                              audio=runtime_info(args.audio_python, ["numpy", "soundfile"])),
                worker_interpreters=dict(asr=str(args.asr_python), audio=str(args.audio_python)),
                parameters=dict(device="cpu", compute_type="int8", cpu_threads=THREADS, num_workers=1,
                                language=None if args.language == "auto" else args.language,
                                beam_size=5, vad_filter=False, word_timestamps=True,
                                condition_on_previous_text=False, initial_prompt=None, hotwords=None,
                                line_recovery_threshold=.6),
                tool_sha256=sha(Path(__file__).resolve()), gpu_used=False)


def fence(text: str) -> str:
    marker = "`" * max(3, max((len(s) + 1 for s in re.findall(r"`+", text)), default=3))
    return f"{marker}text\n{text}\n{marker}"


def markdown(result: dict) -> str:
    lines = ["# CPU lyric QA", "", f"Execution: {result['status']}.", "",
             "ASR is a proxy. No subjective quality, singer identity, or listening approval is inferred.", ""]
    audio = result.get("native_audio")
    if audio:
        lines += [f"Native audio: {audio['duration_seconds']:.3f} s, {audio['sample_rate']} Hz, "
                  f"{audio['channels']} channels; peak {audio['native_sample_peak']:.6f}; "
                  f"nonfinite samples {audio['nonfinite_samples']}.", ""]
    comparison = result.get("comparison")
    if comparison:
        edits = comparison["word_edits"]
        lines += [f"ASR normalized text exact match: {comparison['asr_normalized_text_exact_match']}.",
                  f"Word edits S/D/I: {edits['substitution']}/{edits['deletion']}/{edits['insertion']}. "
                  f"Character similarity: {comparison['normalized_character_similarity']:.3f}; "
                  f"word coverage: {comparison['word_coverage']:.3f}.",
                  f"Approximate lines: {comparison['approximate_lines_recovered']}/{comparison['lines_total']} "
                  "under a global, nonreusing word alignment; approximate recovery is not exact fidelity.", "",
                  "Target:", fence("\n".join(result["target_lines"])), "",
                  "Unprompted transcript:", fence(result["transcript"]["text"]), ""]
    if result.get("error"):
        lines += ["Error:", fence(result["error"]), ""]
    return "\n".join(lines)


def execute(args) -> dict:
    request = make_request(args)
    key = identity(request)
    directory = args.out_dir
    directory.mkdir(parents=True, exist_ok=True)
    with output_lock(directory / "qa.lock"):
        marker = directory / "qa_manifest.json"
        if marker.exists():
            previous = read_json(marker)
            if previous["request_identity"] != key:
                raise ValueError("out-dir already contains different inputs/runtime/settings; choose a new out-dir")
            expected = previous.get("outputs", {})
            if previous.get("status") == "complete" and expected and all(
                    (directory / name).is_file() and sha(directory / name) == digest for name, digest in expected.items()):
                return read_json(directory / "qa.json")
        write_json(directory / "request.json", request)
        (directory / "lyrics_input.txt").write_bytes(args.lyrics.read_bytes())
        (directory / "target_lyrics.txt").write_text("\n".join(request["lyrics"]["target_lines"]) + "\n", encoding="utf-8")
        state = dict(status="running", request_identity=key, started_at=now(), outputs={})
        write_json(marker, state)
        result = dict(schema_version=VERSION, status="running", request_identity=key, gpu_used=False,
                      asr_is_proxy=True, subjective_quality_score=None, singer_identity_score=None,
                      listening_approved=False, target_lines=request["lyrics"]["target_lines"],
                      request_file=str(directory / "request.json"))
        try:
            result["native_audio"] = run_worker(args.audio_python, "audio", dict(audio=str(args.audio)), directory)
            if not result["native_audio"]["valid"]:
                raise ValueError("; ".join(result["native_audio"]["errors"]))
            result["transcript"] = run_worker(args.asr_python, "asr", dict(audio=str(args.audio),
                                                model=str(args.model), language=request["parameters"]["language"]), directory)
            # A producer editing a take during QA must not leave a valid-looking checkpoint.
            if sha(args.audio) != request["audio"]["sha256"] or sha(args.lyrics) != request["lyrics"]["sha256"]:
                raise ValueError("Audio or lyrics changed during QA; use a new output directory")
            if any(sha(args.model / name) != digest for name, digest in request["model"]["files"].items()):
                raise ValueError("ASR model changed during QA")
            result["comparison"] = compare_text(result["target_lines"], result["transcript"]["text"])
            result["status"] = "complete"
        except Exception as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
        result["ended_at"] = now()
        write_json(directory / "qa.json", result)
        (directory / "QA_SUMMARY.md").write_text(markdown(result), encoding="utf-8")
        artifacts = ["qa.json", "QA_SUMMARY.md", "request.json", "lyrics_input.txt", "target_lyrics.txt", "audio.json", "transcript.json"]
        state.update(status=result["status"], ended_at=result["ended_at"],
                     outputs={name: sha(directory / name) for name in artifacts if (directory / name).is_file()})
        write_json(marker, state)
        return result


def parser():
    cli = argparse.ArgumentParser(description="Local unprompted CPU lyric QA; no perceptual/identity score")
    cli.add_argument("--audio", type=Path, required=True)
    cli.add_argument("--lyrics", type=Path, required=True)
    cli.add_argument("--out-dir", type=Path, required=True)
    cli.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    cli.add_argument("--asr-python", type=Path, default=DEFAULT_ASR)
    cli.add_argument("--audio-python", type=Path, default=DEFAULT_AUDIO)
    cli.add_argument("--language", default="en", help="ISO language code, or auto; never target-word conditioning")
    return cli


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--_worker":
        cli = argparse.ArgumentParser()
        cli.add_argument("--_worker", choices=("audio", "asr"), required=True)
        cli.add_argument("--request-file", type=Path, required=True)
        cli.add_argument("--output-file", type=Path, required=True)
        args = cli.parse_args(argv)
        request = read_json(args.request_file)
        data = native_audio(Path(request["audio"])) if args._worker == "audio" else transcribe(request)
        write_json(args.output_file, data)
        return 0
    args = parser().parse_args(argv)
    for name in ("audio", "lyrics", "out_dir", "model", "asr_python", "audio_python"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        result = execute(args)
        print(json.dumps(dict(status=result["status"], report=str(args.out_dir / "QA_SUMMARY.md"),
                              comparison=result.get("comparison"), error=result.get("error")), ensure_ascii=False))
        return 0 if result["status"] == "complete" else 1
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"music_qa: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
