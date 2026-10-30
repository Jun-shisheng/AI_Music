"""Measure a YuE2 symbolic plan (ABC) as numbers: rhythm density and register.

Why this exists (2026-10-16): the user's first complaint about the delivered
epic battle batch was not about timbre or mix -- it was that a whole song runs
on *one* rhythm and *one* register. In the named reference (HOYO-MiX "Wildfire")
the verse is slow and word-sparse and the chorus is dense and higher; ours
"从头到尾都是一个节奏，音调变化也不大". Prompting cannot fix that: the
2026-10-15 A/B proved a rewritten caption does not move the measured arc at all.

The plan, however, is symbolic -- every vocal note onset, its duration and its
pitch are written down before a single sample is rendered. So both properties
are directly measurable here, and (unlike the audio) directly editable, by
`yue2_plan_edit.py`-style transformers. This tool is the *measurement* half of
that loop; without it, "make the verse sparser and the chorus denser" is an
opinion rather than a target.

What the numbers mean, per section:

  onsets     vocal note events (each one is roughly a sung syllable/word)
  onsets/s   density of the words -- THE number the complaint is about
  rest%      share of the section where the voice is silent (breathing room)
  lo/hi      lowest and highest vocal pitch, as MIDI numbers and as a range in
             semitones; the centre of that range is the section's register
  leaps%     share of note-to-note intervals wider than a whole tone (a stepwise
             melody reads as smooth, a leapy one as declamatory)

Read across sections it answers "is the verse sparser than the chorus, and does
the chorus sit higher" without listening to anything.

Usage:
  # one plan
  python -I tools/generation/plan_metrics.py outputs/.../01_x_c1_artifacts/score.abc

  # every plan under a batch folder, one summary row each
  python -I tools/generation/plan_metrics.py --dir outputs/2026-10-15_epic_en

  # machine-readable, for batch ranking
  python -I tools/generation/plan_metrics.py --dir outputs/... --json logs/plan_metrics.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from yue2_plan_edit import _note_midi, key_accidentals, split_plan  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# A plan line is chord symbols in quotes, then notes, barlines, ties and rests.
# Ties (`-`) are deliberately not matched: a tie continues the previous note, it
# is not a new onset, and counting it would inflate the density of exactly the
# long held notes that make a chorus feel big.
_CHORD_RE = re.compile(r'"[^"]*"')
_TOKEN_RE = re.compile(
    r"([zxZ])(\d*(?:/\d*)?)"                          # 1 rest, 2 length
    r"|(\^|_|=)?([A-Ga-g])([',]*)(\d*(?:/\d*)?)(-?)"  # 3..7 note: acc letter octave length tie
)


def _length_units(text: str) -> float:
    """ABC note/rest length in units of the plan's default note length `L:`.

    `d`/empty = 1 unit, `d2` = 2, `d3/2` = 1.5, `d/` and `d/2` = 0.5.
    """
    if not text:
        return 1.0
    match = re.fullmatch(r"(\d*)(?:/(\d*))?", text)
    if not match:
        return 1.0
    number = int(match.group(1)) if match.group(1) else 1
    if "/" not in text:
        return float(number)
    denominator = int(match.group(2)) if match.group(2) else 2
    return number / denominator


def plan_timing(abc: str) -> tuple[float, float]:
    """(seconds per `L:` unit, units per bar) from the plan header.

    `Q:1/q=bpm` says a 1/q note lasts 60/bpm s; `L:1/lden` says one unit is a
    1/lden note, i.e. (1/lden)*(4*60/bpm) s. `M:num/den` says a bar holds `num`
    notes of value 1/den, i.e. num*lden/den units.
    """
    meter_num, meter_den = 4, 4
    lden, q_den, bpm = 4, 4, 120.0
    for raw in abc.splitlines():
        line = raw.strip()
        if line.startswith("M:"):
            match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", line[2:].strip())
            if match:
                meter_num, meter_den = int(match.group(1)), int(match.group(2))
        elif line.startswith("L:"):
            match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", line[2:].strip())
            if match:
                lden = int(match.group(2))
        elif line.startswith("Q:"):
            match = re.search(r"(\d+)\s*/\s*(\d+)\s*=\s*(\d+(?:\.\d+)?)",
                              line[2:])
            if match:
                q_den, bpm = int(match.group(2)), float(match.group(3))
    if bpm <= 0 or lden <= 0 or meter_den <= 0:
        return 0.0, 0.0
    sec_per_unit = (1.0 / lden) * (4.0 * 60.0 / bpm)
    units_per_bar = meter_num * lden / meter_den
    return sec_per_unit, units_per_bar


def voice_lines(body: list[str], voice: str | None = "Vocal") -> list[str]:
    """Bar lines of one voice inside a section (`None` = whichever is there)."""
    lines: list[str] = []
    capture = False
    for raw in body:
        stripped = raw.strip()
        if stripped.startswith("V:"):
            capture = voice is None or stripped[2:].strip().startswith(voice)
            continue
        if capture and stripped:
            lines.append(stripped)
    return lines


def measure_section(name: str, body: list[str], key: dict[str, int],
                    sec_per_unit: float) -> dict:
    """Density, silence share and register of one section's vocal line.

    A purely instrumental section has no `V: Vocal` line at all. It still has a
    length -- so fall back to whatever voice is written and measure the section
    for timing only. Returning 0 s there would shift every later boundary in
    `word_rate.py`, which derives section times from these durations.
    """
    lines = voice_lines(body)
    is_vocal = bool(lines)
    if not is_vocal:
        lines = voice_lines(body, voice=None)
    onsets = 0
    rests_units = 0.0
    total_units = 0.0
    bars = 0
    tied = False                 # the previous note ended with `-`
    pitches: list[int] = []
    for line in lines:
        bars += line.count("|")
        clean = _CHORD_RE.sub("", line)
        for match in _TOKEN_RE.finditer(clean):
            if match.group(1):                       # rest
                units = _length_units(match.group(2))
                rests_units += units
                total_units += units
                tied = False                         # a rest breaks a tie
                continue
            units = _length_units(match.group(6))
            total_units += units
            tied_now = bool(match.group(7))
            if tied or not is_vocal:
                # A tie across a barline (`d8-|d8-|d8|`) is ONE sustained note,
                # not three articulations -- and sustained notes are exactly what
                # a chorus is made of, so counting them would invert the very
                # comparison this tool exists to make.
                tied = tied_now
                continue
            onsets += 1
            pitches.append(_note_midi(match.group(4), match.group(3) or "",
                                      match.group(5), key))
            tied = tied_now
    seconds = total_units * sec_per_unit
    intervals = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
    return {
        "section": name.lstrip("% ").strip(),
        "bars": bars,
        "seconds": round(seconds, 2),
        "onsets": onsets,
        "onsets_per_s": round(onsets / seconds, 3) if seconds > 0 else 0.0,
        "rest_share": round(rests_units / total_units, 3) if total_units else 0.0,
        "lo_midi": min(pitches) if pitches else None,
        "hi_midi": max(pitches) if pitches else None,
        "range_semitones": (max(pitches) - min(pitches)) if pitches else None,
        "centre_midi": round(sum(pitches) / len(pitches), 1) if pitches else None,
        "leap_share": (round(sum(1 for i in intervals if i > 2) / len(intervals), 3)
                       if intervals else 0.0),
    }


def measure_plan(abc: str) -> list[dict]:
    """One metric row per `% section` marker in the plan."""
    header, sections = split_plan(abc)
    key_name = next((ln[2:].strip() for ln in header if ln.startswith("K:")), "C")
    key = key_accidentals(key_name)
    sec_per_unit, _ = plan_timing(abc)
    return [measure_section(name, body, key, sec_per_unit)
            for name, body in sections]


def summarise(rows: list[dict]) -> list[dict]:
    """Mean density / register per section type, in plan order of first sight."""
    order: list[str] = []
    for row in rows:
        if row["section"] not in order:
            order.append(row["section"])
    out = []
    for name in order:
        group = [r for r in rows if r["section"] == name]
        centres = [r["centre_midi"] for r in group if r["centre_midi"] is not None]
        out.append({
            "section": name,
            "occurrences": len(group),
            "onsets_per_s": round(sum(r["onsets_per_s"] for r in group) / len(group), 3),
            "rest_share": round(sum(r["rest_share"] for r in group) / len(group), 3),
            "centre_midi": round(sum(centres) / len(centres), 1) if centres else None,
            "range_semitones": max((r["range_semitones"] or 0) for r in group),
        })
    return out


def print_plan(label: str, rows: list[dict]) -> None:
    print(f"\n== {label}")
    print(f"{'section':<14}{'bars':>5}{'sec':>7}{'onsets':>7}{'ons/s':>7}"
          f"{'rest%':>7}{'lo':>5}{'hi':>5}{'range':>7}{'leaps%':>8}")
    for row in rows:
        print(f"{row['section']:<14}{row['bars']:>5}{row['seconds']:>7.1f}"
              f"{row['onsets']:>7}{row['onsets_per_s']:>7.2f}"
              f"{row['rest_share'] * 100:>7.0f}"
              f"{row['lo_midi'] if row['lo_midi'] is not None else '-':>5}"
              f"{row['hi_midi'] if row['hi_midi'] is not None else '-':>5}"
              f"{row['range_semitones'] if row['range_semitones'] is not None else '-':>7}"
              f"{row['leap_share'] * 100:>8.0f}")
    print(f"{'-- mean by type':<14}")
    for row in summarise(rows):
        centre = row["centre_midi"] if row["centre_midi"] is not None else "-"
        print(f"{row['section']:<14}{'':>5}{'':>7}{'x' + str(row['occurrences']):>7}"
              f"{row['onsets_per_s']:>7.2f}{row['rest_share'] * 100:>7.0f}"
              f"{'':>5}{'':>5}{'range ' + str(row['range_semitones']):>7}"
              f"{'centre ' + str(centre):>8}")


def plans_in(folder: Path) -> list[Path]:
    return sorted(folder.rglob("score.abc"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plans", nargs="*", type=Path, help="score.abc file(s)")
    ap.add_argument("--dir", type=Path, default=None,
                    help="batch folder; every score.abc below it is measured")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    targets = list(args.plans)
    if args.dir:
        targets += plans_in(args.dir)
    if not targets:
        ap.error("no plans given (positional or --dir)")

    payload = {}
    for path in targets:
        rows = measure_plan(path.read_text(encoding="utf-8-sig"))
        if not rows:
            print(f"\n== {path.name}: no section markers")
            continue
        label = path.parent.name if path.name == "score.abc" else path.name
        print_plan(label, rows)
        payload[str(path)] = {"sections": rows, "by_type": summarise(rows)}

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\njson -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
