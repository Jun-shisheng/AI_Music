"""Rewrite the *shape* of a plan's vocal line: how dense the words are.

Why this exists (2026-10-16): the user's first complaint was that a whole song
runs at one rhythm -- "前期和高潮不只是音调的区别，歌手歌词的密度也不一样：
前期明显缓慢，高潮更加集中". Measured on the 2026-10-15 delivery, the chorus is
1.14x the verse (and on several titles *sparser* than the verse); the reference
behaviour is 1.5-2.5x. Prompting cannot fix this -- a rewritten caption moved
nothing (2026-10-15 A/B) -- but the plan is symbolic, so the density can be
written down directly and the renderer follows it (`trim_intro` proved that in
production: cutting the intro bars moved the first sung word from 30.0 s to
3.1 s).

Two operators, both acting on the `V: Vocal` voice only:

  --reduce SECTION:FACTOR   keep 1 onset of every FACTOR (the dropped ones stop
                            being notes and extend the note before them), so the
                            words spread out and the line is held. For verses.
  --split SECTION:MAX_UNITS re-articulate every note longer than MAX_UNITS into
                            equal repeats of the *same pitch*, so a held note
                            becomes words on every beat. For choruses.

What the operators are allowed to change is deliberately narrow:

  * only the `V: Vocal` lines inside matching sections; the `V: Ins` voice, the
    chord symbols and the header are byte-for-byte untouched;
  * pitch material is never rewritten -- `--split` repeats the pitch it finds,
    `--reduce` keeps the surviving pitches exactly, and neither invents notes;
  * every bar keeps its exact total duration (a bar that comes back short or
    long would drift the whole song's tempo) -- checked on every rewrite;
  * ties (`-`) are preserved unless the note is split, and no operator crosses a
    rest with a hold.

Register (the "高的时候不够高" half of the same complaint) stays in
`yue2_plan_edit.py --section ... --semitones ...`, which already transposes a
section correctly against the key signature.

Usage:
  P=vendor/ACE-Step-1.5/.venv/Scripts/python.exe
  $P -I tools/generation/plan_shape.py --file <score.abc> \
      --reduce verse:2 --split chorus:2 --out shaped.abc --report
"""
from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_metrics import (_length_units, measure_plan,  # noqa: E402
                          print_plan, voice_lines)
from yue2_plan_edit import (_transpose_line,  # noqa: E402
                              key_accidentals, split_plan)

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

VOCAL_LINE = "V: Vocal"

_CHORD_RE = re.compile(r'"[^"]*"')
# The chord alternative must come FIRST: a chord symbol like "D#m" contains the
# letter D, and parsing it as a note would (a) invent an onset and (b) let a
# dropped note merge its duration into the symbol, writing "D5#m" and eating the
# real note after it. Chords are kept verbatim as 'other' items.
_TOKEN_RE = re.compile(
    r'("[^"]*")'                                      # 1 chord symbol
    r"|([zxZ])(\d*(?:/\d*)?)"                         # 2,3 rest
    r"|(\^+|_+|=)?([A-Ga-g])([',]*)(\d*(?:/\d*)?)(-?)"  # 4..8 note (^^.. is real)
)


@dataclasses.dataclass
class Item:
    """One token of a bar: a note, a rest, or anything else kept verbatim."""

    kind: str                 # 'note' | 'rest' | 'other'
    units: float = 0.0
    tie: bool = False
    base: str = ""            # note spelling without its length (e.g. '^f' or "c'")
    raw: str = ""             # 'other' payload, e.g. a chord symbol

    def render(self) -> str:
        if self.kind == "other":
            return self.raw
        if self.kind == "rest":
            return self.base + format_units(self.units)
        return self.base + format_units(self.units) + ("-" if self.tie else "")


def _fields_of(line: str) -> list[str]:
    """The `|`-separated bar fields of one voice's line, terminator dropped."""
    stripped = line.strip()
    fields = stripped.split("|")
    if fields and fields[-1].strip() == "":
        fields = fields[:-1]
    return fields


def format_units(units: float) -> str:
    """ABC length suffix for `units`; one unit (the default) writes nothing.

    Whole numbers are written as themselves (`d4`), half units as `n/2` --
    `1/2`, `3/2`, `5/2`. Halves never appear on real plans (every length in the
    delivered plans is a whole number of 1/16 units), so this only matters if a
    source plan already used them.
    """
    if units == 1:
        return ""
    if float(units).is_integer():
        return str(int(units))
    if abs(units * 2 - round(units * 2)) < 1e-9:
        return f"{int(round(units * 2))}/2"
    raise ValueError(f"cannot write {units!r} units as an ABC length")


def parse_field(field: str) -> list[Item]:
    """Tokens of one bar field, in order, with anything unmatched preserved."""
    items: list[Item] = []
    pos = 0
    for match in _TOKEN_RE.finditer(field):
        if match.start() > pos:
            items.append(Item(kind="other", raw=field[pos:match.start()]))
        pos = match.end()
        if match.group(1):                                       # chord symbol
            items.append(Item(kind="other", raw=match.group(1)))
            continue
        if match.group(2):                                       # rest
            items.append(Item(kind="rest", units=_length_units(match.group(3)),
                              base=match.group(2)))
            continue
        base = (match.group(4) or "") + match.group(5) + match.group(6)
        items.append(Item(kind="note", units=_length_units(match.group(7)),
                          tie=bool(match.group(8)), base=base))
    if pos < len(field):
        items.append(Item(kind="other", raw=field[pos:]))
    return items


def field_units(items: list[Item]) -> float:
    return sum(i.units for i in items if i.kind in ("note", "rest"))


def split_units(total: float, max_units: float) -> list[float]:
    """Break `total` into as-equal-as-possible pieces of at most `max_units`.

    Largest pieces first, so a 5-unit note with a 2-unit ceiling becomes
    2+2+1 rather than 1+2+2 -- a long note should start on the beat. Integer
    totals are kept integral (these plans only ever write whole lengths), and a
    half-unit total falls back to halves.
    """
    scale = 1 if float(total).is_integer() else 2
    cap = int(round(max_units * scale))
    remaining = int(round(total * scale))
    pieces: list[float] = []
    while remaining > 0:
        count = -(-remaining // cap)             # ceil
        piece = min(-(-remaining // count), cap)
        pieces.append(piece / scale)
        remaining -= piece
    return pieces


def split_bar(items: list[Item], max_units: float) -> list[Item]:
    """Re-articulate every note longer than `max_units` at the same pitch."""
    out: list[Item] = []
    for item in items:
        if item.kind != "note" or item.units <= max_units:
            out.append(item)
            continue
        # A split note is re-sung, so the original's tie is dropped -- the bar
        # still holds exactly as long as it did.
        for piece in split_units(item.units, max_units):
            out.append(Item(kind="note", units=piece, base=item.base))
    return out


def absorb_backwards(items: list[Item], units: float) -> bool:
    """Extend the nearest earlier note by `units` (searching back past rests).

    A dropped word is absorbed by the note before it, so the verse keeps singing
    on one held note instead of falling silent -- that is the "fewer words, held
    longer" shape being asked for. The note grows by exactly the dropped note's
    duration, so the bar's total is untouched; rests between them are only
    searched across, never modified (taking time out of a rest *and* adding it to
    the note would shrink the bar).

    Returns False when the bar has no earlier note, in which case the caller
    writes a rest instead.
    """
    index = len(items) - 1
    while index >= 0:
        item = items[index]
        if item.kind == "note":
            items[index] = dataclasses.replace(item, units=item.units + units)
            return True
        if item.kind == "rest":
            index -= 1                        # hold over the rest, keep its length
            continue
        index -= 1
    return False


def drop_onsets(items: list[Item], factor: int, continuation: bool,
                mode: str = "hold") -> list[Item]:
    """Keep one onset in every `factor`; the others are held or turned silent.

    `mode="hold"` (--reduce) gives the dropped note's time to the note before it,
    which is what makes a verse sing fewer, longer syllables -- the effect the
    user asked for. `mode="rest"` (--thin) leaves a silence of the same length
    instead, so each surviving syllable keeps its own duration and the phrase
    gains breathing room. The hold is the stronger change: measured 2026-10-16,
    holding every second onset in a verse turned 25 s of words into sustained
    vowels and whisper stopped finding any words at all, so `rest` exists as the
    gentler option.

    `continuation` says the first note of this bar is the tail of a note tied in
    from the previous bar: it is not an onset, so it is never counted or dropped
    (it can still be held longer). Notes tied *within* a bar are handled the same
    way by carrying the tie flag through the item list.
    """
    out: list[Item] = []
    seen = 0
    tied = continuation
    for item in items:
        if item.kind == "other":
            out.append(item)
            continue
        if item.kind == "rest":
            out.append(item)
            tied = False                     # a rest breaks a tie
            continue
        onset = not tied
        tied = item.tie
        if onset and seen % factor != 0:
            seen += 1
            if mode == "rest":
                # Dropping a tied note would break the tie into the next bar, so
                # it is kept instead of silently changing that bar's rhythm.
                out.append(Item(kind="rest", units=item.units, base="z")
                           if not item.tie else item)
            elif not absorb_backwards(out, item.units):
                out.append(Item(kind="rest", units=item.units, base="z"))
            continue
        if onset:
            seen += 1
        out.append(item)
    # Rests emptied by the folds above would render as `z0`.
    return [i for i in out if not (i.kind == "rest" and i.units <= 1e-9)]


def reduce_bar(items: list[Item], factor: int,
               continuation: bool) -> list[Item]:
    """`drop_onsets` in hold mode -- the verse-sparsening operator."""
    return drop_onsets(items, factor, continuation, mode="hold")


def thin_bar(items: list[Item], factor: int,
             continuation: bool) -> list[Item]:
    """`drop_onsets` in rest mode -- sparser without stretching syllables."""
    return drop_onsets(items, factor, continuation, mode="rest")


def rewrite_line(line: str, mode: str, amount: float) -> str:
    """Rewrite one `V: Vocal` bar line, bar field by bar field."""
    stripped = line.strip()
    trailing = "|" if stripped.endswith("|") else ""
    fields = _fields_of(stripped)
    rewritten: list[str] = []
    previous_tie = False
    for field in fields:
        items = parse_field(field)
        before = field_units(items)
        chords = [i.raw for i in items if i.raw.startswith('"')]
        if mode == "split":
            new_items = split_bar(items, amount)
        elif mode == "thin":
            new_items = thin_bar(items, int(amount), previous_tie)
        else:
            new_items = reduce_bar(items, int(amount), previous_tie)
        after = field_units(new_items)
        if abs(after - before) > 1e-9:
            raise ValueError(
                f"bar duration changed ({before} -> {after} units) in {field!r}")
        if chords != [i.raw for i in new_items if i.raw.startswith('"')]:
            raise ValueError(f"a chord symbol was altered in {field!r}")
        notes = [i for i in new_items if i.kind == "note"]
        previous_tie = bool(notes and notes[-1].tie)
        rewritten.append("".join(i.render() for i in new_items))
    return "|".join(rewritten) + trailing


def _section_centre(header: list[str], name: str, body: list[str]) -> float | None:
    """Mean vocal pitch of one section, in MIDI, via the established measure."""
    mini = "\n".join(header) + "\n" + name + "\n" + "\n".join(body) + "\n"
    rows = measure_plan(mini)
    return rows[0]["centre_midi"] if rows else None


def section_matches(name: str, target: str) -> bool:
    """Does this `% section` name match an operator target?

    2026-10-26: `chorus` used to match `pre-chorus` by substring, so
    `hook chorus` copied the chorus melody into the pre-chorus and
    `split chorus` densified it.  A pre-chorus is a launch ramp, not the
    hook; exclude it explicitly.
    """
    low = name.lower()
    target = target.lower().strip()
    if target == "chorus":
        return "chorus" in low and "pre" not in low
    return target in low


def repeat_hook(abc: str, target: str = "chorus",
                match_register: bool = True,
                max_tile: float | None = None) -> tuple[str, list[dict]]:
    """Make every matching section sing the first one's melody.

    Why (2026-10-16, docs/arrangement_spec.md §1.6.2): measured with one
    instrument across 10 vocal references, the loud sections of a real song
    resemble each other at a chroma cosine of 0.978 (median) -- a chorus is the
    same hook coming back. Ours measured 0.902, below every reference, because
    our plan writes each chorus a new melody over the same chords. That is what
    "the whole song runs at one rhythm, the pitch barely moves" sounds like: no
    hook to recognise, so nothing reads as a chorus.

    The bar pool is tiled when the target section has a different bar count than
    the source, so a longer final chorus repeats the motif rather than
    truncating it, and every bar keeps its own duration (a tiled bar is copied
    whole, chord symbol included -- a hook is melody *and* harmony).

    Trailing ties are stripped from the copied bars: a bar copied from the
    middle of the source's phrase may end with `-` tying into a bar that the
    target does not have in that position, and a dangling tie is a different
    rhythm. Each bar still holds exactly as long as before, so the song's tempo
    is untouched.

    The lyrics are not part of the plan -- they are fed to the model separately
    -- so a repeated melody still gets the new words the lyric file provides,
    which is how the user's "the final chorus must be new lyrics" survives.

    `match_register` shifts a copy by whole octaves so it lands in the register
    the target section already used. In one of the delivered takes the first
    chorus sat an octave below the later ones, and a straight copy would have
    dropped the final chorus by an octave -- the opposite of the way a final
    chorus moves. Octaves rather than semitones on purpose: an octave keeps the
    mode, so the copied melody and its chords stay in the song's key.
    """
    header, sections = split_plan(abc)
    matches = [i for i, (name, _) in enumerate(sections) if section_matches(name, target)]
    if len(matches) < 2:
        return abc, []
    src_name, src_body = sections[matches[0]]
    pools: dict[str, list[str]] = {}
    for voice in ("Vocal", "Ins"):
        fields: list[str] = []
        for line in voice_lines(src_body, voice):
            fields.extend(_fields_of(line))
        if fields:
            pools[voice] = fields
    src_bars = len(pools.get("Vocal", []))

    src_centre = (_section_centre(header, src_name, src_body)
                  if match_register else None)
    key_name = next((ln[2:].strip() for ln in header if ln.startswith("K:")), "C")
    key = key_accidentals(key_name)

    report: list[dict] = []
    for index in matches[1:]:
        name, body = sections[index]
        target_bars = sum(len(_fields_of(line))
                          for line in voice_lines(body, "Vocal"))
        # 2026-10-26: a 2-bar chorus tiled over a 25-bar section becomes a
        # 12x loop.  Real songs repeat the hook a few times, not a dozen; when
        # the target is much longer than the source, keep the model's own
        # melody instead of tiling.
        if max_tile and src_bars and target_bars > max_tile * src_bars:
            report.append({"section": name.lstrip("% ").strip(), "mode": "hook",
                           "amount": len(matches) - 1, "bars": target_bars,
                           "skipped": True,
                           "reason": (f"target {target_bars} bars > "
                                      f"{max_tile:g} x source {src_bars} bars")})
            continue
        centre = _section_centre(header, name, body) if match_register else None
        new_body: list[str] = []
        voice = "Vocal"
        cursor = 0
        kept_total = 0
        bars_total = 0
        for line in body:
            stripped = line.strip()
            if stripped.startswith("V:"):
                voice = stripped[2:].strip().split()[0] if stripped[2:].strip() else ""
                cursor = 0
                new_body.append(line)
                continue
            pool = pools.get(voice) or pools.get("Vocal")
            if not pool or not stripped:
                new_body.append(line)
                continue
            fields = _fields_of(stripped)
            rebuilt: list[str] = []
            kept = 0
            bars_total += len(fields)
            for field in fields:
                copy = pool[cursor % len(pool)]
                cursor += 1
                # A copied Vocal bar must hold exactly as long as the bar it
                # replaces, or the section's length changes and the song drifts.
                # (Only the Vocal voice: its bars are guaranteed to sum to the
                # meter, while the Ins lines in these plans write whole-bar rests
                # as `Z` -- 1 unit as parsed -- and are already non-conforming,
                # so their "lengths" carry no timing meaning to begin with.)
                if voice == "Vocal" and abs(
                        field_units(parse_field(copy))
                        - field_units(parse_field(field))) > 1e-9:
                    kept += 1
                    kept_total += 1
                    rebuilt.append(field)
                    continue
                rebuilt.append(copy.rstrip("-"))
            trailing = "|" if stripped.endswith("|") else ""
            new_body.append("|".join(rebuilt) + trailing)

        octaves = 0
        if src_centre is not None and centre is not None:
            octaves = int(round((float(centre) - float(src_centre)) / 12))
        if octaves:
            new_body = [ln if ln.strip().startswith("V:")
                        else _transpose_line(ln, octaves * 12, key)
                        for ln in new_body]

        sections[index] = (name, new_body)
        report.append({"section": name.lstrip("% ").strip(), "mode": "hook",
                       "amount": len(matches) - 1, "bars": bars_total,
                       "register_octaves": octaves,
                       "bars_kept_for_length_mismatch": kept_total})

    new_abc = "\n".join(header)
    for name, body in sections:
        new_abc += ("\n" if not new_abc.endswith("\n") else "") + name
        for line in body:
            new_abc += "\n" + line
    return new_abc.rstrip("\n") + "\n", report


def rewrite_section(body: list[str], mode: str, amount: float) -> tuple[list[str], int]:
    """Rewrite the vocal lines of one section; returns (lines, bars touched)."""
    out: list[str] = []
    capturing = False
    touched = 0
    for line in body:
        stripped = line.strip()
        if stripped.startswith("V:"):
            capturing = stripped == VOCAL_LINE
            out.append(line)
            continue
        if capturing and stripped:
            touched += stripped.count("|")
            out.append(rewrite_line(line, mode, amount))
            continue
        out.append(line)
    return out, touched


def apply_shape(abc: str, reduce_specs: list[str], split_specs: list[str],
                thin_specs: list[str] = (),
                hook_specs: list[str] = (),
                hook_max_tile: float | None = None) -> tuple[str, list[dict]]:
    """Apply `section:amount` operators to every matching `% section`."""
    report: list[dict] = []
    # Hook first: copying the first chorus's melody onto the later ones has to
    # happen before the density operators, so every chorus is then densified
    # identically instead of each per its own melody.
    for spec in hook_specs:
        section, _, _ = spec.partition(":")
        abc, hook_report = repeat_hook(abc, section or "chorus",
                                       max_tile=hook_max_tile)
        report.extend(hook_report)
    header, sections = split_plan(abc)
    rules: list[tuple[str, str, float]] = []
    for spec, mode, default in (
            [(s, "reduce", 2) for s in reduce_specs]
            + [(s, "split", 2) for s in split_specs]
            + [(s, "thin", 2) for s in thin_specs]):
        name, _, amount = spec.partition(":")
        rules.append((name, mode, float(amount or default)))

    for index, (name, body) in enumerate(sections):
        for target, mode, amount in rules:
            if target and not section_matches(name, target):
                continue
            new_body, bars = rewrite_section(body, mode, amount)
            if bars:
                sections[index] = (name, new_body)
                report.append({"section": name.lstrip("% ").strip(),
                               "mode": mode, "amount": amount, "bars": bars})

    new_abc = "\n".join(header)
    for name, body in sections:
        new_abc += ("\n" if not new_abc.endswith("\n") else "") + name
        for line in body:
            new_abc += "\n" + line
    return new_abc.rstrip("\n") + "\n", report


def parse_op(spec: str) -> tuple[str, str, float]:
    """`"split chorus:2"` -> `("split", "chorus", 2.0)`, for batch plumbing.

    The batch tools have to carry an operator through a command line, a driver
    and a subprocess, so the whole thing has to fit in one shell-safe token.
    """
    mode, _, target = spec.strip().partition(" ")
    if mode not in ("reduce", "split", "thin", "hook"):
        raise ValueError(f"unknown operator {mode!r} in {spec!r}")
    section, _, amount = target.partition(":")
    return mode, section, float(amount or 2)


def apply_ops(abc: str, ops: list[str],
              hook_max_tile: float | None = None) -> tuple[str, list[dict]]:
    """Apply several CLI-style ops in one pass, e.g.
    ["hook chorus", "split chorus:2"] -- the hook is copied before the density
    operators run, whatever order the ops are given in."""
    buckets: dict[str, list[str]] = {"reduce": [], "split": [], "thin": [],
                                     "hook": []}
    for spec in ops:
        mode, section, amount = parse_op(spec)
        buckets[mode].append(f"{section}:{amount:g}")
    return apply_shape(abc, buckets["reduce"], buckets["split"], buckets["thin"],
                       buckets["hook"], hook_max_tile=hook_max_tile)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, required=True, help="source score.abc")
    ap.add_argument("--reduce", action="append", default=[], metavar="SECTION:FACTOR",
                    help="spread the words out by HOLDING the previous note "
                         "(verses), e.g. verse:2 -- measured 2026-10-16 to be too "
                         "strong at 2: the verse's words became sustained vowels")
    ap.add_argument("--thin", action="append", default=[], metavar="SECTION:FACTOR",
                    help="spread the words out by inserting RESTS instead, so each "
                         "syllable keeps its length, e.g. thin verse:2")
    ap.add_argument("--split", action="append", default=[], metavar="SECTION:MAX_UNITS",
                    help="re-articulate held notes (choruses), e.g. chorus:2")
    ap.add_argument("--hook", action="append", default=[], metavar="SECTION",
                    help="make every later SECTION sing the first one's melody, "
                         "so the chorus is a hook that returns (measured: our "
                         "choruses resemble each other at chroma 0.902 against a "
                         "reference median of 0.978), e.g. hook chorus")
    ap.add_argument("--hook-max-tile", type=float, default=None,
                    help="skip hook copying when the target section is more than "
                         "this many times the source length (prevents 10x loops)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", action="store_true",
                    help="print the density table before and after")
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()

    abc = args.file.read_text(encoding="utf-8-sig")
    new_abc, report = apply_shape(abc, args.reduce, args.split, args.thin,
                                  args.hook, hook_max_tile=args.hook_max_tile)
    if not report:
        print("no section matched the given rules; nothing changed")
        return 1
    for entry in report:
        print(f"{entry['section']}: {entry['mode']} {entry['amount']:g} "
              f"-> {entry['bars']} bars rewritten")
    if args.report:
        print_plan("before", measure_plan(abc))
        print_plan("after", measure_plan(new_abc))
    if args.out:
        args.out.write_text(new_abc, encoding="utf-8")
        print(f"\nwrote {args.out}")
    if args.dump:
        print(new_abc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
