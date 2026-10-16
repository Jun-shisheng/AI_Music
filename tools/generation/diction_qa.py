"""Diction QA: how much of the intended lyric is actually intelligible?

This project has repeatedly shipped takes whose words were slurred ("吐字不清")
and only caught it by ear. This tool transcribes a generated take with
faster-whisper and scores it against the lyric that was fed to the model.

What the numbers mean:
  ratio     character-level similarity (difflib) between the transcript and the
            reference lyric, after stripping section tags, punctuation and
            whitespace. ~1.0 = the words came back verbatim; <0.3 usually means
            the model sang something else entirely (or sang nothing).
  lines     how many reference lines are >=60 % recovered. A take can have a
            decent overall ratio while dropping whole lines, which is what a
            broken phrase sounds like.
  coverage  length of the transcript relative to the reference; much less than
            1.0 means parts were never sung (the model emits EOS early).
  onset     when the first word is sung (`first_vocal_s`). Added 2026-10-15 for
            the user's "前奏太长" complaint: they want the vocal in within a few
            seconds, or an intro of at most 10-20 s, and the model decides that
            on its own -- so it has to be measured and gated. Word timestamps
            come from the same transcription, so this costs nothing extra.
  tail      `last_vocal_s`, the end of the last sung word (how much outro is
            instrumental).

Writes full UTF-8 text to a report file, because the transcript itself must not
be trusted to a GBK console.

Usage:
  python tools/generation/diction_qa.py --lyrics inputs/battle_test/02_battle_vocal_lyrics.txt \
      --out logs/diction_battle.md outputs/battle_test/battle_vocal_ace_v1_c1.wav ...
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / \
    "PR_AIMusic_Extension" / "models" / "faster-whisper-small"

TAG_RE = re.compile(r"^\s*\[[^\]]*\]\s*$")
PUNCT_RE = re.compile(r"[\s，。、！？；：,.!?;:\"'“”‘’()（）\-—~～…]+")


def clean_lyrics(text: str) -> list[str]:
    """Reference lines with section tags and punctuation removed."""
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or TAG_RE.match(raw):
            continue
        line = PUNCT_RE.sub("", raw)
        if line:
            lines.append(line)
    return lines


def clean_transcript(text: str) -> str:
    return PUNCT_RE.sub("", text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+", type=Path)
    ap.add_argument("--lyrics", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=MODEL_DIR)
    ap.add_argument("--language", default="zh",
                    help="ISO code, or 'auto' to let whisper detect (use for "
                         "mixed-language songs)")
    ap.add_argument("--out", type=Path, default=None,
                    help="markdown report (full UTF-8 text)")
    ap.add_argument("--json", type=Path, default=None,
                    help="machine-readable scores for batch ranking")
    args = ap.parse_args()

    ref_lines = clean_lyrics(args.lyrics.read_text(encoding="utf-8"))
    ref_all = "".join(ref_lines)

    from faster_whisper import WhisperModel
    model = WhisperModel(str(args.model), device="cpu", compute_type="int8")
    language = None if args.language in ("auto", "") else args.language

    report = []
    scores = []
    print(f"{'file':<36}{'ratio':>7}{'lines':>9}{'cov':>6}{'onset':>7}{'tail':>7}  language")
    for wav in args.wavs:
        segments, info = model.transcribe(str(wav), language=language,
                                          beam_size=5, vad_filter=False,
                                          word_timestamps=True)
        seg_list = list(segments)
        text = "".join(s.text for s in seg_list).strip()
        hyp = clean_transcript(text)
        # First/last sung word -> how long the instrumental intro runs before the
        # vocal enters, and how much of the ending is empty.
        word_times = [(w.start, w.end) for s in seg_list
                      for w in (s.words or [])
                      if (w.word or "").strip()]
        first_vocal = round(min(t[0] for t in word_times), 2) if word_times else -1.0
        last_vocal = round(max(t[1] for t in word_times), 2) if word_times else -1.0

        ratio = difflib.SequenceMatcher(None, ref_all, hyp).ratio()
        recovered = 0
        for line in ref_lines:
            if len(line) >= 2:
                best = max((difflib.SequenceMatcher(None, line, hyp[i:i + len(line) + 4]).ratio()
                            for i in range(max(len(hyp) - len(line) + 1, 1))), default=0.0)
                if best >= 0.6:
                    recovered += 1
        coverage = len(hyp) / max(len(ref_all), 1)

        print(f"{wav.name[:35]:<36}{ratio:>7.2f}"
              f"{recovered:>4}/{len(ref_lines):<4}{coverage:>6.2f}"
              f"{first_vocal:>7.1f}{last_vocal:>7.1f}  {info.language}")
        scores.append({
            "file": str(wav),
            "ratio": round(ratio, 3),
            "lines_recovered": recovered,
            "lines_total": len(ref_lines),
            "coverage": round(coverage, 3),
            "first_vocal_s": first_vocal,
            "last_vocal_s": last_vocal,
            "language": info.language,
            "transcript": text,
        })
        report.append(
            f"### {wav.name}\n\n"
            f"- ratio `{ratio:.2f}` | lines `{recovered}/{len(ref_lines)}` | "
            f"coverage `{coverage:.2f}` | 首句 `{first_vocal:.1f}s` | "
            f"末句 `{last_vocal:.1f}s` | lang `{info.language}` "
            f"(p={info.language_probability:.2f})\n\n"
            f"**Transcript**\n\n```\n{text}\n```\n\n"
            f"**Reference**\n\n```\n{' '.join(ref_lines)}\n```\n"
        )

    if args.out:
        header = (f"# Diction QA\n\n- reference: `{args.lyrics}`\n"
                  f"- model: `{args.model.name}` (faster-whisper, {args.language}, cpu/int8)\n\n")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(header + "\n".join(report), encoding="utf-8")
        print(f"\nreport -> {args.out}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(scores, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
