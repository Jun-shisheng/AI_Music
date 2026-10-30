"""Prepare Mandarin SoulX SVS metadata from a source singing performance.

Mandarin is encoded as whole syllables with tone (`zh_<pinyin><tone>`), so unlike
the English path there is no CMU dictionary and no phoneme splitting: one hanzi
is one note. Pitch comes from pyin on the source vocal, timing from ASR word
timestamps, and `control=score` means the model reads note_pitch only — the f0
contour is not copied, which is what lets new lyrics keep the melody.

Everything here is approximate until a human listens. No alignment is claimed
as reviewed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import audio
from .svs_score import record

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHONESET = ROOT / "vendor/voice_studio/SoulX-Singer/soulxsinger/utils/phoneme/phone_set.json"
TONE_ORDER = [2, 4, 3, 1, 5]


def load_phoneset(path: Path) -> set[str]:
    return set(json.loads(path.read_text(encoding="utf-8")))


def normalize_syllable(syllable: str) -> str:
    return syllable.replace("ü", "u:").replace("v", "u:")


def line_tokens(chars: str, phoneset: set[str]) -> list[tuple[str, str, str | None]]:
    """Convert a whole line at once so polyphones keep their phrase reading.

    A single hanzi converted alone gets the dictionary default (吭 -> keng1), which
    is wrong inside 引吭高歌 (hang2). Returns (char, token, substitution note).
    """
    from pypinyin import Style, lazy_pinyin

    readings = lazy_pinyin(chars, style=Style.TONE3, neutral_tone_with_five=True,
                           errors=lambda item: list(item))
    if len(readings) != len(chars):
        raise ValueError(f"Pinyin count {len(readings)} does not match {len(chars)} characters")
    return [(char, *reading_token(reading, char, phoneset)) for char, reading in zip(chars, readings)]


def reading_token(reading: str, char: str, phoneset: set[str]) -> tuple[str, str | None]:
    """Return (token, substitution note). Raises when no honest token exists."""
    match = re.fullmatch(r"([a-zü]+)([1-5])", reading, re.IGNORECASE)
    if not match:
        raise ValueError(f"{char!r} has no tone reading: {reading!r}")
    syllable, tone = normalize_syllable(match.group(1).lower()), match.group(2)
    token = f"zh_{syllable}{tone}"
    if token in phoneset:
        return token, None
    for other in TONE_ORDER:
        alternative = f"zh_{syllable}{other}"
        if alternative in phoneset:
            return alternative, f"{token}->{alternative} (tone not in phoneset)"
    for swap in (("u:", "u"), ("u", "u:")):
        alternative = f"zh_{syllable.replace(*swap)}{tone}"
        if alternative in phoneset:
            return alternative, f"{token}->{alternative} (umlaut form)"
    raise ValueError(f"No phoneset token for {char!r} ({reading}); refusing to guess")


def asr(args) -> dict:
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      OPENBLAS_NUM_THREADS="2")
    from faster_whisper import WhisperModel

    model = WhisperModel(str(args.asr_model), device="cpu", compute_type="int8",
                         cpu_threads=2, num_workers=1)
    segments, info = model.transcribe(str(args.vocal), language="zh", beam_size=5,
                                      vad_filter=False, word_timestamps=True,
                                      condition_on_previous_text=False)
    rows = []
    for segment in segments:
        words = [dict(word=w.word, start=w.start, end=w.end, probability=w.probability)
                 for w in segment.words or []]
        rows.append(dict(start=segment.start, end=segment.end, text=segment.text,
                         avg_logprob=segment.avg_logprob, no_speech_prob=segment.no_speech_prob,
                         words=words))
    result = dict(status="automatic_asr_unreviewed", device="cpu", gpu_used=False,
                  vocal=str(args.vocal.resolve()), vocal_sha256=audio.digest(args.vocal),
                  model=str(args.asr_model.resolve()), language=info.language,
                  segments=rows, transcript="".join(r["text"].strip() for r in rows),
                  warning="Sung-word ASR mishears lyrics. Characters and timings must be listened to before they define a score.")
    audio.atomic_json(args.out, result)
    return result


def characters(transcript: dict, start: float, end: float) -> tuple[list[dict], list[dict]]:
    """Split ASR words into characters, dividing each word's span evenly."""
    chars: list[dict] = []
    dropped: list[dict] = []
    for segment in transcript["segments"]:
        for word in segment["words"]:
            text = re.sub(r"[^一-鿿]", "", word["word"])
            if not text:
                dropped.append(dict(word=word["word"], reason="no_hanzi"))
                continue
            a, b = max(start, float(word["start"])), min(end, float(word["end"]))
            if b - a < 0.04:
                dropped.append(dict(word=word["word"], reason="outside_window_or_too_short",
                                    asr=[word["start"], word["end"]]))
                continue
            if b <= start or a >= end:
                continue
            # A word straddling the window edge loses its clipped characters.
            span = b - a
            step = span / len(text)
            for index, char in enumerate(text):
                a_i, b_i = a + step * index, a + step * (index + 1)
                chars.append(dict(char=char, start=a_i, end=b_i, probability=word["probability"]))
    chars.sort(key=lambda item: item["start"])
    return chars, dropped


def contour(vocal: Path, start: float, end: float) -> np.ndarray:
    import librosa

    data, rate = librosa.load(str(vocal), sr=24000, mono=True, offset=start, duration=end - start)
    hop = 480  # 50 Hz grid, same as the official prompt contour
    f0, voiced, _ = librosa.pyin(data, fmin=float(librosa.note_to_hz("C3")),
                                 fmax=float(librosa.note_to_hz("C6")), sr=rate, hop_length=hop,
                                 frame_length=2048, fill_na=0.0)
    return np.nan_to_num(f0, nan=0.0)


def midi(frequency: float) -> int:
    return int(round(69 + 12 * np.log2(frequency / 440.0)))


def prepare(args) -> dict:
    transcript = json.loads(args.asr_json.read_text(encoding="utf-8"))
    if transcript.get("vocal_sha256") != audio.digest(args.vocal):
        raise ValueError("ASR transcript belongs to a different vocal file")
    phoneset = load_phoneset(args.phoneset)
    start, end = args.start, args.end
    if end <= start:
        raise ValueError("--end must be after --start")
    chars, dropped = characters(transcript, start, end)
    if not chars:
        raise ValueError("No sung characters inside the window")
    f0 = contour(args.vocal, start, end)
    if abs(len(f0) / 50 - (end - start)) > .05:
        raise ValueError("Contour is not on the expected 50 Hz grid")

    entries: list[dict] = []
    substitutions: list[dict] = []
    unvoiced: list[dict] = []
    cursor = start

    def rest(until: float) -> None:
        nonlocal cursor
        if until - cursor > .08:
            entries.append(dict(start=cursor, end=until, text="<SP>", phoneme="<SP>", pitch=0, type=1))
            cursor = until

    tokens = line_tokens("".join(item["char"] for item in chars), phoneset)
    for item, (_, token, note) in zip(chars, tokens):
        begin, finish = max(cursor, item["start"]), min(end, item["end"])
        if finish - begin < .04:
            dropped.append(dict(word=item["char"], reason="overlaps_previous_character"))
            continue
        rest(begin)
        if note:
            substitutions.append(dict(char=item["char"], change=note))
        frames = f0[int(begin * 50):max(int(begin * 50) + 1, int(finish * 50))]
        voiced_frames = frames[frames > 0]
        if len(voiced_frames) < 2:
            unvoiced.append(dict(char=item["char"], start=begin, end=finish))
            pitch = entries[-1]["pitch"] if entries and entries[-1]["pitch"] else 60
        else:
            pitch = midi(float(np.median(voiced_frames)))
        entries.append(dict(start=begin, end=finish, text=item["char"], phoneme=token,
                            pitch=pitch, type=2))
        cursor = finish
    rest(end)
    if cursor < end:
        entries.append(dict(start=cursor, end=end, text="<SP>", phoneme="<SP>", pitch=0, type=1))

    # Round to microseconds; durations must cover the window within the 80 ms contract.
    items = [dict(start=round(e["start"], 6), end=round(e["end"], 6), text=e["text"],
                  phoneme=e["phoneme"], pitch=e["pitch"], type=e["type"]) for e in entries]
    meta = record(items, end - start, args.name)
    meta["time"] = [0, round((end - start) * 1000)]
    meta["index"] = args.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.json"
    audio.atomic_json(out, [meta])
    report = dict(status="approximate_alignment", window_seconds=[start, end],
                  characters=len([e for e in items if e["type"] == 2]), rests=len([e for e in items if e["type"] == 1]),
                  dropped=dropped, tone_substitutions=substitutions, unvoiced_estimates=unvoiced,
                  lyric_text="".join(e["text"] for e in items if e["type"] == 2),
                  note_range=[min(e["pitch"] for e in items if e["type"] == 2),
                              max(e["pitch"] for e in items if e["type"] == 2)],
                  metadata=str(out.resolve()), metadata_sha256=audio.digest(out),
                  human_reviewed=False,
                  warning="ASR timing, pyin pitch and tone substitution are all unreviewed; listen before delivery.")
    audio.atomic_json(args.out_dir / f"{args.name}_report.json", report)
    return report


def validate_cpu(args) -> dict:
    sys.path.insert(0, str(args.soulx_root.resolve()))
    sys.path.insert(0, str(args.soulx_root.resolve() / "cli"))
    from soulxsinger.utils.data_processor import DataProcessor

    meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))[0]
    processor = DataProcessor(hop_size=480, sample_rate=24000, phoneset_path=str(args.phoneset),
                              device="cpu")
    item = processor.process(dict(meta))
    note_type = item["note_type"][0].tolist()
    mel2note = item["mel2note"][0].tolist()
    mapped = [n for n in set(mel2note) if n >= 0]
    silent = [n for n in mapped if not any(mel2note[f] == n for f in range(len(mel2note))
                                           if note_type[n] != 1 and f < len(mel2note))]
    report = dict(frames=len(mel2note), notes=len(note_type),
                  pad_frames=sum(1 for n in mel2note if n < 0),
                  zero_phone_notes=len(silent), human_reviewed=False,
                  warning="CPU mapping feasibility only; it does not prove diction or rhythm.")
    audio.atomic_json(Path(args.out), report)
    if report["pad_frames"] or report["zero_phone_notes"]:
        raise SystemExit(json.dumps(report, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("asr", help="Chinese word timestamps, CPU only")
    a.add_argument("--vocal", type=Path, required=True)
    a.add_argument("--asr-model", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("prepare", help="Build target metadata for one window")
    p.add_argument("--vocal", type=Path, required=True)
    p.add_argument("--asr-json", type=Path, required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--name", default="zh_target")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--phoneset", type=Path, default=DEFAULT_PHONESET)
    v = sub.add_parser("validate-cpu", help="Official DataProcessor feasibility check")
    v.add_argument("--metadata", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--phoneset", type=Path, default=DEFAULT_PHONESET)
    v.add_argument("--soulx-root", type=Path, default=ROOT / "vendor/voice_studio/SoulX-Singer")
    args = parser.parse_args()
    result = {"asr": asr, "prepare": prepare, "validate-cpu": validate_cpu}[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
