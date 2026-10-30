"""A strict editor for the small ABC dialect emitted by our YuE2 plans.

Instrument voice experiments do not edit melody, lyrics, key, tempo or harmony.
The engine still decides actual timbres: an ``Ins`` note is not a drum command.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import re


@dataclass
class Chunk:
    vocal: list[str]
    ins: list[str]


@dataclass
class Section:
    marker: str
    chunks: list[Chunk]


TOKEN = re.compile(r'"[^"\n]*"|(?:\^+|_+|=)?[A-Ga-g][,\x27]*\d*(?:/\d*)?-?|[zxZ]\d*(?:/\d*)?')
NOTE = re.compile(r'((?:\^+|_+|=)?[A-Ga-g][,\x27]*)(\d*(?:/\d*)?)(-?)$')
CHORD = re.compile(r'"([^"\n]*)"')


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def length(suffix: str) -> Fraction:
    if not suffix:
        return Fraction(1)
    if "/" in suffix:
        n, d = suffix.split("/")
        return Fraction(int(n or 1), int(d or 2))
    return Fraction(int(suffix))


def suffix(units: Fraction) -> str:
    if units == 1:
        return ""
    if units.denominator == 1:
        return str(units.numerator)
    return f"{units.numerator}/{units.denominator}"


def timing(header: list[str]) -> dict:
    fields = {s.split(":", 1)[0]: s.split(":", 1)[1].strip()
              for s in header if ":" in s}
    meter = Fraction(fields.get("M", "4/4"))
    default = Fraction(fields.get("L", "1/8"))
    match = re.fullmatch(r'(\d+/\d+)\s*=\s*(\d+(?:\.\d+)?)', fields.get("Q", "1/4=120"))
    if not match:
        raise ValueError("Need an explicit Q:1/4=120-style tempo")
    beat, bpm = Fraction(match[1]), float(match[2])
    if meter <= 0 or default <= 0 or bpm <= 0:
        raise ValueError("Meter, default note length and tempo must be positive")
    return {"units_per_bar": meter / default,
            "seconds_per_bar": float(meter / beat) * 60 / bpm,
            "tempo_bpm": bpm, "quarter_bpm": bpm * float(beat / Fraction(1, 4))}


def parse(abc: str) -> tuple[list[str], list[Section]]:
    header, sections = [], []
    current = None
    voice = None
    for raw in abc.splitlines():
        line = raw.strip()
        if line.startswith("%"):
            if not re.fullmatch(r'%\s*[A-Za-z][A-Za-z0-9 _-]*', line):
                raise ValueError(f"Unsupported section marker: {line}")
            current = Section(raw, [])
            sections.append(current)
            voice = None
        elif current is None:
            header.append(raw)
        elif line == "V: Vocal":
            current.chunks.append(Chunk([], []))
            voice = "vocal"
        elif line == "V: Ins":
            if not current.chunks:
                raise ValueError("Ins appears before Vocal; this editor needs paired voices")
            voice = "ins"
        elif line:
            if voice is None or not current.chunks:
                raise ValueError(f"Unsupported body line: {line}")
            getattr(current.chunks[-1], voice).append(raw)
    if not sections or any(not s.chunks for s in sections):
        raise ValueError("Need nonempty % sections with paired Vocal/Ins voices")
    if any(not c.vocal or not c.ins for s in sections for c in s.chunks):
        raise ValueError("Every chunk must have both Vocal and Ins")
    timing(header)
    return header, sections


def serialize(header: list[str], sections: list[Section]) -> str:
    lines = list(header)
    for sec in sections:
        lines.append(sec.marker)
        for chunk in sec.chunks:
            lines += ["V: Vocal", *chunk.vocal, "V: Ins", *chunk.ins]
    return "\n".join(lines).rstrip() + "\n"


def strip_music_chords(abc: str) -> tuple[str, dict]:
    """Remove chord symbols from music only for YuE2 cot=melody.

    Quoted instrument names in headers remain byte-for-byte intact. Note,
    rest, tie and bar text is otherwise untouched, including both voices.
    """
    header, sections = parse(abc)
    before = [line for section in sections for chunk in section.chunks
              for line in chunk.vocal + chunk.ins]
    removed = sum(len(CHORD.findall(line)) for line in before)
    clock = timing(header)
    counts_before = [len(bars(getattr(chunk, voice), clock["units_per_bar"]))
                     for section in sections for chunk in section.chunks for voice in ("vocal", "ins")]
    for section in sections:
        for chunk in section.chunks:
            chunk.vocal = [CHORD.sub("", line) for line in chunk.vocal]
            chunk.ins = [CHORD.sub("", line) for line in chunk.ins]
    after = [line for section in sections for chunk in section.chunks
             for line in chunk.vocal + chunk.ins]
    if after != [CHORD.sub("", line) for line in before]:
        raise AssertionError("cot=melody conversion changed non-chord music text")
    counts_after = [len(bars(getattr(chunk, voice), clock["units_per_bar"]))
                    for section in sections for chunk in section.chunks for voice in ("vocal", "ins")]
    if counts_before != counts_after:
        raise AssertionError("cot=melody conversion changed voice durations")
    return serialize(header, sections), {
        "conversion": "music_chord_symbols_removed_for_cot_melody",
        "removed_chord_symbols": removed, "header_unchanged": True,
        "note_rest_tie_bar_text_unchanged": True, "voice_durations_unchanged": True}


def bars(lines: list[str], units_per_bar: Fraction) -> list[str]:
    """Expand multi-measure Z4 rests while rejecting unrecognised notation."""
    out = []
    for raw in lines:
        fields = raw.strip().split("|")
        if fields[-1].strip():
            raise ValueError("Every music line must end in |")
        for field in fields[:-1]:
            field = field.strip()
            multi = re.fullmatch(r'Z(\d*)', field)
            if multi:
                out.extend(["z" + suffix(units_per_bar)] * int(multi[1] or 1))
                continue
            tokens = list(TOKEN.finditer(field))
            residual = TOKEN.sub("", field)
            if residual.strip():
                raise ValueError(f"Unsupported ABC syntax {residual!r} in {field!r}")
            total = Fraction(0)
            for tok in tokens:
                text = tok[0]
                if text.startswith('"'):
                    continue
                if text.startswith("Z"):
                    raise ValueError("Z multirest must occupy its own field")
                if text[0] in "zx":
                    total += length(text[1:])
                else:
                    total += length(NOTE.fullmatch(text)[2])
            if total != units_per_bar:
                raise ValueError(f"Bar duration {total} != expected {units_per_bar}: {field}")
            out.append(field)
    return out


def vocal_identity(header: list[str], sections: list[Section]) -> str:
    return sha("\n".join(header + [v for sec in sections for c in sec.chunks for v in c.vocal]))


def section_name(marker: str) -> str:
    return marker.lstrip("% ").strip().lower()


def select_excerpt(abc: str, start_section="verse", section_count=2,
                   selectors: list[str] | None = None) -> tuple[str, dict]:
    header, sections = parse(abc)
    if selectors:
        selected = []
        for selector in selectors:
            name, occurrence = selector.rsplit(":", 1) if ":" in selector else (selector, "1")
            matches = [(i, s) for i, s in enumerate(sections) if section_name(s.marker) == name.lower()]
            n = int(occurrence)
            if n < 1 or n > len(matches):
                raise ValueError(f"No section {selector}")
            selected.append(matches[n - 1])
        if any(b[0] != a[0] + 1 for a, b in zip(selected, selected[1:])):
            raise ValueError("Excerpt sections must be consecutive and in source order")
    else:
        start = next((i for i, s in enumerate(sections)
                      if section_name(s.marker) == start_section.lower()), None)
        if start is None or section_count < 1 or start + section_count > len(sections):
            raise ValueError("Requested consecutive excerpt is missing")
        selected = list(enumerate(sections))[start:start + section_count]
    clock = timing(header)
    offsets, position = {}, 0
    for index, section in enumerate(sections):
        count = sum(len(bars(c.vocal, clock["units_per_bar"])) for c in section.chunks)
        offsets[index] = (position, count)
        position += count
    result = [s for _, s in selected]
    count = sum(offsets[i][1] for i, _ in selected)
    warnings = []
    for sec in result:
        for n, chunk in enumerate(sec.chunks):
            vocal = bars(chunk.vocal, clock["units_per_bar"])
            ins = bars(chunk.ins, clock["units_per_bar"])
            if len(ins) != len(vocal):
                warnings.append(f"{sec.marker} chunk {n+1}: source Ins {len(ins)} bars vs Vocal {len(vocal)}; baseline kept verbatim, edited Ins aligned to Vocal")
    return serialize(header, result), {
        "sections": [s.marker for s in result], "bars": count,
        "first_section_occurrence": sum(section_name(s.marker) == section_name(result[0].marker)
                                         for s in sections[:selected[0][0] + 1]),
        "duration_s": round(count * clock["seconds_per_bar"], 3),
        "source_start_s": round(offsets[selected[0][0]][0] * clock["seconds_per_bar"], 3),
        "vocal_identity": vocal_identity(header, result), "warnings": warnings,
        "note": "Symbolic timing is a prediction; measure rendered audio separately."}


def _root(symbol: str) -> tuple[int, bool]:
    match = re.match(r'^([A-G])([#b]?)(.*)$', symbol)
    if not match:
        raise ValueError(f"Cannot derive accompaniment from chord {symbol!r}")
    root = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[match[1]]
    root += 1 if match[2] == "#" else -1 if match[2] == "b" else 0
    if match[3] not in {"", "m", "maj", "7", "m7", "maj7", "6", "m6", "9", "m9", "maj9", "add9"}:
        raise ValueError(f"Unsupported chord quality {symbol!r}; supply a manual Ins plan")
    minor = match[3].startswith("m") and not match[3].startswith("maj")
    return root % 12, minor


def _spelling(midi: int) -> str:
    names = ("=C", "^C", "=D", "^D", "=E", "=F", "^F", "=G", "^G", "=A", "^A", "=B")
    octave, pc = divmod(midi, 12)
    octave -= 1
    name = names[pc]
    if octave >= 5:
        return name.lower() + "'" * (octave - 5)
    return name + "," * (4 - octave)


def accompaniment(field: str, chord: str, mode: str, units: Fraction, family="arpeggio") -> str:
    """Use chord-root notes, never the reference melody; keep Ins annotations."""
    quoted = "".join(m[0] for m in CHORD.finditer(field))
    if mode == "sparse":
        root, _ = _root(chord)
        return quoted + _spelling(48 + root) + suffix(units / 2) + "z" + suffix(units / 2)
    root, minor = _root(chord)
    if family == "support":
        # A restrained support voice: two quarter-bar chord tones followed by
        # half a bar of space. This is a timing-control hypothesis, not a claim
        # that the learned renderer will obey duration or instrument identity.
        return (quoted + _spelling(48 + root) + suffix(units / 4) +
                _spelling(48 + root + 7) + suffix(units / 4) + "z" + suffix(units / 2))
    tones = [48 + root, 60 + root + (3 if minor else 4), 60 + root + 7, 60 + root]
    # Eight evenly spaced arpeggio articulations; a single Ins voice cannot
    # independently specify kick/snare/strings or guarantee full-band sound.
    return quoted + "".join(_spelling(tones[i % 4]) + suffix(units / 8) for i in range(8))


def resolve_mapping(sections: list[Section], blueprint: dict | None) -> tuple[dict, dict]:
    """Record which reference targets actually control each destination section.

    A source ``section_04`` may map to a destination ``chorus`` without claiming
    that the source itself is a chorus. Unmatched targets remain visible.
    """
    candidates = []
    for index, source in enumerate((blueprint or {}).get("sections", [])):
        target = source.get("target", {})
        explicit = target.get("plan_section")
        label = str(explicit or source.get("name", "")).lower()
        label = re.sub(r'[_ ]\d+$', '', label)
        pattern = target.get("ins_pattern")
        candidates.append({"source_index": index, "source_section": source.get("name"),
                           "source_start_s": source.get("start_s"), "source_end_s": source.get("end_s"),
                           "source_annotation_status": source.get("annotation_status", "unknown"),
                           "source_role_observation_status": source.get("role_observation_status", "unknown"),
                           "target_plan_section": label, "ins_pattern": pattern,
                           "mapping_basis": "explicit_destination_slot" if explicit else "matching_source_label",
                           "proposal_status": target.get("proposal_status", "unknown")})
    selected, matched, unmatched, used = {}, [], [], set()
    for index, section in enumerate(sections):
        name = section_name(section.marker)
        available = [c for c in candidates if c["target_plan_section"] == name and c["ins_pattern"] in {"sparse", "full"}]
        if available:
            entry = available[0]
            selected[index] = entry["ins_pattern"]
            used.add(entry["source_index"])
            matched.append({"plan_section_index": index, "plan_section": name, **entry})
        else:
            fallback = "full" if any(x in name for x in ("chorus", "drop", "climax")) and "pre" not in name else "sparse"
            selected[index] = fallback
            unmatched.append({"plan_section_index": index, "plan_section": name,
                              "ins_pattern": fallback, "provenance": "generic_control",
                              "reason": "no_valid_reference_target_for_destination_section" if blueprint else "no_reference_blueprint"})
    provenance = ("source_blueprint" if matched and not unmatched else
                  "mixed_source_blueprint_and_generic_control" if matched else "generic_control")
    summary = {"provenance": provenance, "matched_plan_section_count": len(matched),
               "unmatched_plan_section_count": len(unmatched),
               "matched_plan_sections": matched, "unmatched_plan_sections": unmatched,
               "unused_blueprint_sections": [c for c in candidates if c["source_index"] not in used],
               "source_blueprint_supplied": blueprint is not None,
               "warning": ("No reference section target matched. Arranged is a generic Ins control experiment, not reference-driven."
                           if not matched else None),
               "interpretation": "Mapping describes destination plan slots, not the musical identity of a source section; all proposals require listening."}
    return selected, summary


def edit_instrumental(abc: str, mode: str, blueprint: dict | None = None, family="arpeggio") -> tuple[str, dict]:
    if mode not in {"sparse", "full", "arranged"}:
        raise ValueError("mode must be sparse, full or arranged")
    if family not in {"arpeggio", "support"}:
        raise ValueError("family must be arpeggio or support")
    header, sections = parse(abc)
    before = vocal_identity(header, sections)
    clock = timing(header)
    changes, previous_chord = [], None
    targets, mapping = resolve_mapping(sections, blueprint)
    for si, sec in enumerate(sections):
        chosen = mode
        name = section_name(sec.marker)
        if mode == "arranged":
            # Blueprint targets only affect the *arranged* arm. The sparse/full
            # arms remain a deliberate stress test of Ins controllability.
            chosen = targets[si]
        for ci, chunk in enumerate(sec.chunks):
            vbars = bars(chunk.vocal, clock["units_per_bar"])
            ibars = bars(chunk.ins, clock["units_per_bar"])
            if any(CHORD.search(field) for field in ibars[len(vbars):]):
                raise ValueError("Extra unmatched Ins bars contain chords; cannot discard harmony safely")
            old = list(chunk.ins)
            new = []
            for bi, field in enumerate(vbars):
                found = CHORD.findall(field)
                if len(found) > 1:
                    raise ValueError("Multiple chord changes per bar need a manual accompaniment plan")
                previous_chord = found[0] if found else previous_chord
                if not previous_chord:
                    raise ValueError("Cannot invent harmony: first retained bar needs a chord")
                new.append(accompaniment(ibars[bi] if bi < len(ibars) else "", previous_chord, chosen, clock["units_per_bar"], family))
            chunk.ins = ["|".join(new) + "|"]
            # Explicit byte and duration checks, not a promise in documentation.
            if len(bars(chunk.ins, clock["units_per_bar"])) != len(vbars):
                raise AssertionError("Instrument voice lost a bar")
            changes.append({"section": name, "chunk": ci + 1, "mode": chosen,
                            "bars": len(vbars), "before": old, "after": chunk.ins,
                            "source_ins_bars": len(ibars), "duration_preserved_against": "Vocal"})
    after = vocal_identity(header, sections)
    if before != after:
        raise AssertionError("Instrument editor changed Vocal/header")
    return serialize(header, sections), {
        "variant": mode, "only_voice_edited": "Ins", "vocal_header_sha256": before,
        "instrument_pattern_family": family,
        "vocal_header_unchanged": True, "chords_in_vocal_unchanged": True,
        "all_instrument_bars_duration_checked": True, "changes": changes,
        "mapping": mapping if mode == "arranged" else {"provenance": "generic_control_stress_test", "reference_targets_applied": False},
        "control_limit": "Ins is one symbolic melody voice; actual instrument identity and drum patterns are not guaranteed."}


def select_lyrics(text: str, selected_markers: list[str], first_occurrence=1) -> str:
    """Select original lyric blocks, including pre-choruses folded into verse."""
    blocks = []
    for match in re.finditer(r'^\[([^\]]+)\][^\n]*\n?', text, re.M):
        blocks.append((match.start(), match.end(), match[1]))
    if not blocks:
        raise ValueError("Lyrics need [Verse]/[Chorus] section markers or --excerpt-lyrics")
    groups, pending = [], []
    for index, (start, _, name) in enumerate(blocks):
        end = blocks[index + 1][0] if index + 1 < len(blocks) else len(text)
        lower = name.lower()
        if "pre" in lower and "chorus" in lower:
            if groups:
                groups[-1][1].append(text[start:end].strip())
            else:
                pending.append(text[start:end].strip())
            continue
        base = "chorus" if "chorus" in lower else re.sub(r'\s+\d+$', '', lower)
        groups.append([base, pending + [text[start:end].strip()]])
        pending = []
    # Match a consecutive subsequence, keeping repeat occurrences in order.
    wanted = [section_name(s) for s in selected_markers]
    occurrence = 0
    for start in range(len(groups)):
        if groups[start][0] == wanted[0]:
            occurrence += 1
        if occurrence != first_occurrence:
            continue
        if [g[0] for g in groups[start:start + len(wanted)]] == wanted:
            return "\n\n".join(part for g in groups[start:start + len(wanted)] for part in g[1]) + "\n"
    raise ValueError("ABC sections do not match consecutive lyric blocks; supply --excerpt-lyrics explicitly")
