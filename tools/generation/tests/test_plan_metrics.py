"""Tests for the plan-shaped measurement (density / register / rests).

2026-10-16: these numbers decide how the plan is edited to give the verse and
the chorus different density and register (docs/arrangement_spec.md §1), so the
parser itself has to be pinned down. Every case below is a property that a
transformer is allowed to assume.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plan_metrics import measure_plan, plan_timing, summarise  # noqa: E402

HEADER = "X:1\nT:\nM:2/4\nL:1/16\nQ:1/4=73\nV: Vocal\nV: Ins\nK:D#m\n"


def plan(*sections: str) -> str:
    """Build a plan from `name::vocal bars::ins bars` triples.

    An empty vocal part emits no `V: Vocal` line at all, which is how a purely
    instrumental section looks in a real plan.
    """
    body = ""
    for section in sections:
        name, vocal, ins = section.split("::")
        body += f"% {name}\n"
        if vocal:
            body += f"V: Vocal\n{vocal}\n"
        body += f"V: Ins\n{ins}\n"
    return HEADER + body


def only(rows: list[dict], section: str) -> dict:
    return next(r for r in rows if r["section"] == section)


def test_timing_from_meter_length_and_tempo():
    # L:1/16 at 73 bpm -> one unit = (1/16)*(4*60/73) s; M:2/4 -> a bar is 8 units.
    sec_per_unit, units_per_bar = plan_timing(HEADER)
    assert round(sec_per_unit, 4) == round(240 / (16 * 73), 4)
    assert units_per_bar == 8


def test_density_counts_onsets_not_tied_continuations():
    # `d2d2d2ef-|` is 5 onsets (d2 d2 d2 e f) filling one 8-unit bar; the tie `-`
    # continues the previous note and must not be counted.
    rows = measure_plan(plan("verse::d2d2d2ef-|::Z|"))
    assert only(rows, "verse")["onsets"] == 5
    assert only(rows, "verse")["bars"] == 1


def test_tie_across_a_barline_is_one_note_not_three():
    # Real plans hold notes this way (`d8-|d8-|d8|`), and a held note is one
    # articulation: counting bars as onsets would make a chorus look denser than
    # it sounds and hide the very contrast this tool measures.
    tied = only(measure_plan(plan("chorus::d8-|d8-|d8|::Z|")), "chorus")
    untied = only(measure_plan(plan("chorus::d8|d8|d8|::Z|")), "chorus")
    assert tied["onsets"] == 1
    assert untied["onsets"] == 3
    assert tied["bars"] == untied["bars"] == 3
    assert tied["seconds"] == untied["seconds"]


def test_rest_breaks_a_tie():
    rows = measure_plan(plan("verse::d4-|z2d2|::Z|"))
    assert only(rows, "verse")["onsets"] == 2


def test_section_length_comes_from_durations_including_rests():
    # One bar of 8 units = 8 * 0.2055 s; a wholly silent section is still long.
    rows = measure_plan(plan("verse::d2d2d2ef-|::Z|", "intro::z8|::Z|"))
    verse, intro = only(rows, "verse"), only(rows, "intro")
    assert verse["seconds"] == round(8 * 240 / (16 * 73), 2)
    assert intro["onsets"] == 0
    assert intro["rest_share"] == 1.0
    assert intro["onsets_per_s"] == 0.0


def test_rest_share_separates_breathing_room_from_word_density():
    dense = measure_plan(plan("chorus::d2d2d2d2|::Z|"))[0]
    sparse = measure_plan(plan("verse::d4z4|::Z|"))[0]
    assert dense["rest_share"] == 0.0
    assert sparse["rest_share"] == 0.5
    assert dense["onsets_per_s"] > sparse["onsets_per_s"]


def test_register_tracks_key_signature_and_octave():
    # Upper case = octave 4 (MIDI 60 + pitch class); K:D#m carries 6 sharps
    # (F C G D A E), so an unmarked F sounds F# (66) and an unmarked c sounds
    # C# (73), not F natural (65) / C natural (72).
    sharp = measure_plan(plan("verse::F8|::Z|"))[0]
    natural = measure_plan(plan("verse::F8|::Z|").replace("K:D#m", "K:C"))[0]
    assert sharp["lo_midi"] == 66
    assert natural["lo_midi"] == 65
    # Lower case sits an octave up (C#5 = 73), and a single pitch spans nothing.
    two_octaves = measure_plan(plan("verse::F8|::Z|", "chorus::c8|::Z|"))
    assert two_octaves[0]["lo_midi"] == 66
    assert two_octaves[1]["hi_midi"] == 73
    assert two_octaves[1]["range_semitones"] == 0


def test_leap_share_marks_wide_intervals():
    stepwise = measure_plan(plan("verse::d2e2d2c2|::Z|"))[0]
    leaping = measure_plan(plan("verse::d2a2d2A2|::Z|"))[0]
    assert stepwise["leap_share"] == 0.0
    assert leaping["leap_share"] == 1.0


def test_summary_averages_by_section_type():
    rows = measure_plan(plan("verse::d4z4|::Z|", "chorus::d2d2d2d2|::Z|",
                             "verse::d4z4|::Z|"))
    by_type = {r["section"]: r for r in summarise(rows)}
    assert by_type["verse"]["occurrences"] == 2
    assert by_type["chorus"]["occurrences"] == 1
    assert by_type["verse"]["rest_share"] == 0.5
    assert by_type["chorus"]["onsets_per_s"] > by_type["verse"]["onsets_per_s"]


def test_instrumental_only_section_keeps_its_length():
    # No `V: Vocal` line: onsets stay 0, but the section must still occupy its
    # bars, otherwise every later section boundary shifts (see word_rate.py).
    rows = measure_plan(plan("interlude::::d2d2d2d2|"))
    row = only(rows, "interlude")
    assert row["onsets"] == 0
    assert row["bars"] == 1
    assert row["seconds"] == round(8 * 240 / (16 * 73), 2)
