"""Edit a YuE2 symbolic plan (ABC notation) before it is rendered.

Why this exists: YuE2 plans every song as ABC notation first and can render from
a *supplied* plan (`SongRequest.abc` -- `pipeline.plan()` returns the supplied
text verbatim instead of sampling a new one). That makes the plan the one place
where song structure can be changed deterministically instead of hoped for.

The problem it solves (2026-10-15): the user wants the vocal to start within a
few seconds, or at most 10-20 s. Measured on the nine delivered 2026-10-14
tracks with whisper word timestamps, the first sung word landed at 15.5-32.3 s,
median 30.0 s -- because the model plans an instrumental intro of 18 bars
before the first lyric. Prompting did not change it; the plan does. Trimming
those bars out is exact, and it costs one cheap plan pass (~60 s) rather than
re-rolling a whole song (5-11 min).

The intro is trimmed in whole *chunks* (a `V: Vocal` line plus the `V: Ins`
line that sounds under the same bars), never bar-by-bar: the two voices of a
chunk must stay aligned, and dropping both together keeps the plan consistent.

Anything else in the plan is left byte-for-byte alone.

Usage (offline inspection of a saved plan):
  python tools/generation/yue2_plan_edit.py \
      --file outputs/.../01_x_c1_artifacts/score.abc --keep-bars 4
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

VOCAL_LINE = "V: Vocal"
INS_LINE = "V: Ins"


def is_section_header(line: str) -> bool:
    """`% intro`, `% verse`, `% chorus` ... are the plan's structure markers."""
    return line.startswith("%")


def split_plan(abc: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Split ABC into (header lines, [(section name, body lines), ...])."""
    lines = abc.splitlines()
    header: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    for line in lines:
        if is_section_header(line):
            sections.append((line.strip(), []))
        elif sections:
            sections[-1][1].append(line)
        else:
            header.append(line)
    return header, sections


def section_bars(body: list[str]) -> int:
    """Bars in a section, counted on the Vocal voice (falls back to any voice).

    One bar is one `|`-terminated field in the ABC body.
    """
    counts: list[int] = []
    capture = False
    for line in body:
        stripped = line.strip()
        if stripped.startswith("V:"):
            capture = stripped == VOCAL_LINE
            continue
        if capture and stripped:
            counts.append(stripped.count("|"))
    if not counts:                      # no vocal voice: use whatever is there
        counts = [line.strip().count("|") for line in body
                  if line.strip() and not line.strip().startswith("V:")]
    return sum(counts)


def chunk_spans(body: list[str]) -> list[tuple[int, int, int]]:
    """(start, end, bars) per `V: Vocal` line and the block that sounds with it.

    `bars` counts the Vocal line only: the `V: Ins` line under the same bars is
    simultaneous, not additional time. Counting both was a real bug -- it
    reported the 18-bar intro of the 2026-10-14 plan as 36 bars.
    """
    starts = [i for i, line in enumerate(body) if line.strip() == VOCAL_LINE]
    spans: list[tuple[int, int, int]] = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(body)
        bars = sum(body[j].strip().count("|") for j in range(start + 1, end)
                   if body[j].strip() and not body[j].strip().startswith("V:"))
        # A chunk normally holds exactly one bar line per voice; when it does
        # not (two Ins lines), fall back to the largest single line.
        lines = [body[j].strip().count("|") for j in range(start + 1, end)
                 if body[j].strip() and not body[j].strip().startswith("V:")]
        spans.append((start, end, max(lines) if lines else bars))
    return spans


def _drop_leading_bars(line: str, count: int) -> str | None:
    """Remove the first `count` `|`-terminated fields from an ABC bar line.

    Returns None when the line does not have that many bars, so the caller can
    fall back to dropping the whole chunk rather than desynchronising the voices.
    """
    fields = line.split("|")
    if fields and fields[-1].strip() == "":
        fields = fields[:-1]                  # trailing separator
    if len(fields) < count:
        return None
    return "|".join(fields[count:]) + "|"


def seconds_per_bar(abc: str) -> float:
    """Bar length in seconds from the plan's meter and tempo.

    `M:a/b` says a bar holds `a` notes of value 1/b; `Q:1/q=bpm` says a 1/q note
    lasts 60/bpm seconds. So one bar = a * (60/bpm) * (q/b). Falls back to
    4/4 at 120 bpm if the header is missing either field.
    """
    num, den = 4, 4
    q_den, bpm = 4, 120.0
    for line in abc.splitlines():
        stripped = line.strip()
        if stripped.startswith("M:"):
            match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", stripped[2:].strip())
            if match:
                num, den = int(match.group(1)), int(match.group(2))
        elif stripped.startswith("Q:"):
            match = re.search(r"(\d+)\s*/\s*(\d+)\s*=\s*(\d+(?:\.\d+)?)", stripped)
            if match:
                q_den = int(match.group(2))
                bpm = float(match.group(3))
    if bpm <= 0:
        return 0.0
    return num * (60.0 / bpm) * (q_den / den)


def trim_intro(abc: str, keep_bars: int = 2) -> tuple[str, dict]:
    """Drop leading intro bars until only `keep_bars` remain.

    Whole chunks are removed while the target is still out of reach, then the
    boundary chunk is sliced bar-by-bar on *every* voice at once so the vocal
    and instrumental lines stay aligned. A plan whose intro is already short is
    returned unchanged.

    Returns the new ABC and a report (bars before/after, seconds saved).
    """
    header, sections = split_plan(abc)
    if not sections:
        return abc, {"changed": False, "reason": "no section markers"}
    keep_bars = max(keep_bars, 1)   # never emit an empty intro section

    per_bar = seconds_per_bar(abc)
    bars_before = 0
    dropped = 0
    out_sections: list[tuple[str, list[str]]] = []

    for name, body in sections:
        if not name.startswith("% intro"):
            out_sections.append((name, body))
            continue
        bars_before = section_bars(body)
        if bars_before <= keep_bars:
            out_sections.append((name, body))
            continue

        want = bars_before - keep_bars
        kept: list[str] = []
        for start, end, bars in chunk_spans(body):
            if want <= 0:
                kept.extend(body[start:end])
                continue
            if want >= bars:
                want -= bars
                dropped += bars
                continue
            # Boundary chunk: slice the same number of bars off every voice.
            sliced: list[str] = []
            ok = True
            for line in body[start:end]:
                stripped = line.strip()
                if not stripped or stripped.startswith("V:"):
                    sliced.append(line)
                    continue
                new_line = _drop_leading_bars(stripped, want)
                if new_line is None:
                    ok = False
                    break
                sliced.append(new_line)
            if ok:
                dropped += want
                want = 0
                kept.extend(sliced)
            else:
                # Voices disagree on bar count -- drop the whole chunk instead
                # of leaving them out of phase.
                dropped += bars
                want = max(want - bars, 0)
        out_sections.append((name, kept))

    new_abc = "\n".join(header)
    for name, body in out_sections:
        new_abc += ("\n" if not new_abc.endswith("\n") else "") + name
        for line in body:
            new_abc += "\n" + line
    new_abc = new_abc.rstrip("\n") + "\n"

    info = {
        "changed": dropped > 0,
        "intro_bars_before": bars_before,
        "intro_bars_after": bars_before - dropped,
        "dropped_bars": dropped,
        "dropped_seconds": round(dropped * per_bar, 1),
        "seconds_per_bar": round(per_bar, 3),
    }
    return new_abc, info


# --- pitch lift for a section (2026-10-15) ----------------------------------
#
# The user's second complaint was "高的时候不够高". In every plan inspected the
# final chorus is a *literal* repeat of the first chorus's chords and melody --
# e.g. the 10-14 take 01 has "D#m"d4a4|"C#"gfe2c2c2| in both `% chorus` blocks.
# A real step-up modulation (Starfall goes Dm -> D for its final chorus) is
# therefore something the plan has to be told to do.
#
# ABC spelling rules this has to respect:
#   * octave: `C`..`B` uppercase = octave 4, `c`..`b` lowercase = octave 5,
#     `,` drops an octave, `'` raises one;
#   * an unmarked note takes the accidentals of the `K:` key signature, so a
#     correct transposition must read the key signature and then write an
#     explicit `^`/`_`/`=` only where the result differs from what the key
#     already implies.

_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_PC = {0: ("C", 0), 1: ("C", 1), 2: ("D", 0), 3: ("D", 1), 4: ("E", 0),
             5: ("F", 0), 6: ("F", 1), 7: ("G", 0), 8: ("G", 1), 9: ("A", 0),
             10: ("A", 1), 11: ("B", 0)}
_FLAT_PC = {0: ("C", 0), 1: ("D", -1), 2: ("D", 0), 3: ("E", -1), 4: ("E", 0),
            5: ("F", 0), 6: ("G", -1), 7: ("G", 0), 8: ("A", -1), 9: ("A", 0),
            10: ("B", -1), 11: ("B", 0)}

_MAJOR_SHARP = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7}
_MAJOR_FLAT = {"F": 1, "Bb": 2, "Eb": 3, "Ab": 4, "Db": 5, "Gb": 6, "Cb": 7}
_MINOR_SHARP = {"A": 0, "E": 1, "B": 2, "F#": 3, "C#": 4, "G#": 5, "D#": 6, "A#": 7}
_MINOR_FLAT = {"D": 1, "G": 2, "C": 3, "F": 4, "Bb": 5, "Eb": 6, "Ab": 7}
_SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
_FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]

_NOTE_RE = re.compile(r"(\^|_|=)?([A-Ga-g])([',]*)(\d*(?:/\d*)?)")
_CHORD_RE = re.compile(r'"([^"]*)"')
_CHORD_ROOT_RE = re.compile(r"^([A-G][#b]?)(.*)$", re.DOTALL)


def key_accidentals(key: str) -> dict[str, int]:
    """Letter -> accidental implied by a `K:` value ('D#m' -> 6 sharps)."""
    match = re.fullmatch(r"([A-G][#b]?)(m?)", key.strip())
    if not match:
        return {}
    root, minor = match.group(1), bool(match.group(2))
    table = _MINOR_SHARP if minor else _MAJOR_SHARP
    flats = _MINOR_FLAT if minor else _MAJOR_FLAT
    if root in table:
        return {letter: 1 for letter in _SHARP_ORDER[: table[root]]}
    if root in flats:
        return {letter: -1 for letter in _FLAT_ORDER[: flats[root]]}
    return {}


def _note_midi(letter: str, accidental: str, marks: str,
               key: dict[str, int]) -> int:
    """Absolute MIDI pitch of an ABC note token."""
    pc = _LETTER_PC[letter.upper()]
    if accidental == "^":
        pc += 1
    elif accidental == "_":
        pc -= 1
    elif accidental == "=":
        pass
    else:
        pc += key.get(letter.upper(), 0)
    octave = 4 if letter.isupper() else 5
    octave += marks.count("'") - marks.count(",")
    return 12 * (octave + 1) + pc


def _midi_to_abc_note(midi: int, key: dict[str, int], direction: int,
                      prefer_letter: str | None = None,
                      always_explicit: bool = False) -> str:
    """ABC token for a MIDI pitch.

    `prefer_letter` keeps the source spelling when the pitch allows it (E sharp
    stays `^e` instead of being respelled as `=f`), which matters because
    transposing by zero must be a no-op. `always_explicit` is for chord symbols,
    which are not governed by the key signature and must state their accidental.
    """
    pc = midi % 12
    letter, offset = (_SHARP_PC if direction >= 0 else _FLAT_PC)[pc]
    if prefer_letter:
        alt = (pc - _LETTER_PC[prefer_letter.upper()] + 6) % 12 - 6
        if -1 <= alt <= 1:
            letter, offset = prefer_letter.upper(), alt
    implied = key.get(letter, 0)
    if always_explicit:
        accidental = "" if offset == 0 else ("^" if offset > 0 else "_")
    elif offset == implied:
        accidental = ""
    elif offset == 0:
        accidental = "="
    elif offset > 0:
        accidental = "^"
    else:
        accidental = "_"
    octave = midi // 12 - 1
    if octave >= 5:
        body = letter.lower() + "'" * (octave - 5)
    else:
        body = letter + "," * (4 - octave)
    return f"{accidental}{body}"


def _transpose_chord(chord: str, semitones: int, key: dict[str, int]) -> str:
    """Transpose a chord root, keeping the model's own `#`/`b` spelling.

    Chord symbols are absolute and carry no octave, and the plans write them as
    "D#m"/"Bmaj7" rather than with ABC accidentals -- so emit `#`/`b` here, and
    never a bare letter with the accidental implied (that turned "D#m" into "Dm").
    """
    match = _CHORD_ROOT_RE.match(chord)
    if not match:
        return chord
    root, suffix = match.group(1), match.group(2)
    letter, accidental = root[0], root[1:]
    pc = (_LETTER_PC[letter] + {"#": 1, "b": -1}.get(accidental, 0)
          + semitones) % 12
    new_letter, offset = (_SHARP_PC if semitones >= 0 else _FLAT_PC)[pc]
    return new_letter + {1: "#", -1: "b", 0: ""}[offset] + suffix


def _transpose_line(line: str, semitones: int, key: dict[str, int]) -> str:
    """Transpose chords and notes on one ABC line, leaving everything else."""
    out: list[str] = []
    pos = 0
    for match in _CHORD_RE.finditer(line):
        out.append(_substitute_notes(line[pos:match.start()], semitones, key))
        out.append('"' + _transpose_chord(match.group(1), semitones, key) + '"')
        pos = match.end()
    out.append(_substitute_notes(line[pos:], semitones, key))
    return "".join(out)


def _substitute_notes(text: str, semitones: int, key: dict[str, int]) -> str:
    def repl(match: re.Match) -> str:
        letter = match.group(2)
        midi = _note_midi(letter, match.group(1) or "", match.group(3), key)
        note = _midi_to_abc_note(midi + semitones, key, semitones,
                                 prefer_letter=letter)
        return note + match.group(4)

    return _NOTE_RE.sub(repl, text)


def transpose_section(abc: str, section: str = "chorus", semitones: int = 2,
                      occurrence: int = -1) -> tuple[str, dict]:
    """Shift one section (default: the last chorus) up by `semitones`.

    Returns the new plan and a report. The key signature is left alone: the
    transposed section carries explicit accidentals, which is exactly what a
    modulation looks like in ABC.
    """
    header, sections = split_plan(abc)
    key_name = next((ln[2:].strip() for ln in header if ln.startswith("K:")), "C")
    key = key_accidentals(key_name)

    matches = [i for i, (name, _) in enumerate(sections) if section in name]
    if not matches:
        return abc, {"changed": False, "reason": f"no section matching {section!r}"}
    idx = matches[occurrence]
    name, body = sections[idx]
    new_body = [
        line if line.strip().startswith("V:")
        else _transpose_line(line, semitones, key)
        for line in body
    ]
    sections[idx] = (name, new_body)

    new_abc = "\n".join(header)
    for sec_name, sec_body in sections:
        new_abc += ("\n" if not new_abc.endswith("\n") else "") + sec_name
        for line in sec_body:
            new_abc += "\n" + line
    new_abc = new_abc.rstrip("\n") + "\n"

    return new_abc, {
        "changed": True,
        "section": name,
        "occurrence": occurrence,
        "semitones": semitones,
        "key": key_name,
        "key_accidentals": key,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--keep-bars", type=int, default=2)
    ap.add_argument("--section", default=None,
                    help="transpose the last section whose marker contains this "
                         "name, e.g. 'chorus' for the final chorus")
    ap.add_argument("--semitones", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump", action="store_true", help="print the edited plan")
    args = ap.parse_args()

    abc = args.file.read_text(encoding="utf-8")
    new_abc, info = trim_intro(abc, keep_bars=args.keep_bars)
    print(f"{args.file.name}: {info}")
    if args.section:
        new_abc, lift_info = transpose_section(
            new_abc, section=args.section, semitones=args.semitones)
        print(f"{args.file.name}: {lift_info}")
    if args.out:
        args.out.write_text(new_abc, encoding="utf-8")
        print(f"wrote {args.out}")
    if args.dump:
        print(new_abc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
