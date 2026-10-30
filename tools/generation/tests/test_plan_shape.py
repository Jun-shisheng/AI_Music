"""Tests for the density rewrite of a plan's vocal line.

2026-10-16: this is the operator that gives the verse and the chorus different
word density (docs/arrangement_spec.md §1.5). It edits a file the renderer
follows literally, so the bar-length and chord-symbol invariants below are not
style preferences -- breaking one corrupts the song (a mangled chord symbol was
the real first bug: `"D#m"` parsed its `D` as a note and swallowed the note
after it).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plan_metrics import measure_plan  # noqa: E402
from plan_shape import (absorb_backwards, apply_ops, apply_shape,  # noqa: E402
                        format_units, parse_field, reduce_bar, repeat_hook,
                        rewrite_line, split_units)

HEADER = "X:1\nT:\nM:2/4\nL:1/16\nQ:1/4=76\nV: Vocal\nV: Ins\nK:D#m\n"


def line_of(vocal: str, ins: str = "Z4|") -> str:
    return vocal


def rewrite(vocal: str, mode: str, amount: float) -> str:
    return rewrite_line(vocal, mode, amount)


def units_of(field: str) -> float:
    return sum(i.units for i in parse_field(field) if i.kind in ("note", "rest"))


def test_reduce_halves_the_onsets_and_keeps_the_bar_length():
    # 5 onsets in one 8-unit bar -> the dropped ones extend the note before them.
    out = rewrite("d2d2d2ef-|", "reduce", 2)
    assert out == "d4d3f-|"
    assert units_of(out.rstrip("|")) == units_of("d2d2d2ef-")


def test_reduce_with_factor_one_changes_nothing():
    vocal = '"D#m"d4a4|"D#m"a4d2ef-|'
    assert rewrite(vocal, "reduce", 1) == vocal


def test_reduce_holds_through_a_rest_instead_of_falling_silent():
    # `g4z3d`: the trailing d is dropped, and its time is held by the g (the
    # rest keeps its own length) -- the verse should sing on, not go quiet.
    assert rewrite("g4z3d|", "reduce", 2) == "g5z3|"
    assert units_of("g5z3") == 8


def test_reduce_absorbs_into_the_previous_note_not_the_next():
    # `d2a2` with factor 2 drops the a; the d absorbs it and the bar still sums
    # to 8 units. Absorbing forwards instead would move the surviving pitch
    # earlier in the bar, which is a rhythmic change, not a density change.
    assert rewrite("d2a2d2a2|", "reduce", 2) == "d4d4|"


def test_the_first_onset_of_a_bar_is_always_kept():
    # Anchoring each bar to its first attack is what keeps the beat: with the
    # keep rule `seen % factor == 0` the first onset always survives, so a
    # dropped note always has a note before it inside the bar.
    assert rewrite("a2d2d2d2|", "reduce", 2) == "a4d4|"
    assert rewrite("a2d2d2d2|", "reduce", 3) == "a6d2|"


def test_absorb_backwards_has_no_note_to_extend_in_an_empty_bar():
    # Defensive contract: only reachable if the keep rule above ever changes.
    assert absorb_backwards([], 2) is False


def test_continuation_after_a_tie_is_not_an_onset():
    # The second bar is the tail of the tied note in the first: it must not be
    # counted (or dropped) as a word of its own.
    rows = measure_plan(HEADER + "% verse\nV: Vocal\nd8-|d8|\nV: Ins\nZ2|\n")
    assert rows[0]["onsets"] == 1
    assert rewrite("d8-|d8|", "reduce", 2) == "d8-|d8|"


def test_thin_replaces_dropped_onsets_with_rests():
    # Keeping 1 onset in 2, but as silence instead of a hold: every surviving
    # syllable keeps its 2 units (0.39 s) where --reduce would stretch it to 8
    # (1.58 s). That stretch is what turned a verse's words into sustained
    # vowels in the 2026-10-16 A/B, which is why this operator exists.
    out = rewrite("d2d2d2d2|", "thin", 2)
    assert out == "d2z2d2z2|"
    assert units_of(out.rstrip("|")) == 8


def test_thin_keeps_the_first_onset_of_each_bar():
    assert rewrite("a2d2d2d2|", "thin", 2) == "a2z2d2z2|"


def test_thin_does_not_drop_a_tied_note():
    # Dropping `d2-` would break the tie into the next bar and change that bar's
    # rhythm, so a tied note is kept even when the count says drop it.
    assert rewrite("d4z2d2-|d2|", "thin", 2) == "d4z2d2-|d2|"


def hook_plan() -> str:
    """A verse plus two choruses with different melodies and bar counts."""
    return (HEADER
            + '% verse\nV: Vocal\n"D#m"d2d2d2d2|\nV: Ins\nZ|\n'
            + '% chorus\nV: Vocal\n"D#m"d4a4|"C#"gfe2c2c2|\nV: Ins\nZ2|\n'
            + '% chorus\nV: Vocal\n"C#"c4g4|\nV: Ins\nZ|\n')


def test_repeat_hook_copies_the_first_chorus_onto_the_later_ones():
    # Measured on the 2026-10-15 delivery, our loud sections resemble each other
    # at chroma 0.902 where 10 vocal references sit at 0.978 -- each chorus is a
    # new melody over the same chords, so nothing reads as a hook.
    out, report = repeat_hook(hook_plan(), "chorus")
    assert [r["section"] for r in report] == ["chorus"]
    assert out.count('"D#m"d4a4|') == 2          # the second chorus now sings it
    assert '"C#"c4g4|' not in out                # its own melody is gone
    assert '"D#m"d2d2d2d2|' in out               # the verse is untouched


def test_repeat_hook_tiles_the_motif_when_bar_counts_differ():
    # A 3-bar target chorus against a 2-bar source gets motif, second bar, motif
    # again -- a longer final statement repeats the hook instead of truncating it.
    three_bars = (HEADER + '% chorus\nV: Vocal\n"D#m"d4a4|"C#"gfe2c2c2|\nV: Ins\nZ2|\n'
                  + '% chorus\nV: Vocal\n"B"d4d4|"B"e4e4|"B"f4f4|\nV: Ins\nZ3|\n')
    out, _ = repeat_hook(three_bars, "chorus")
    bars = out.split("% chorus")[-1].split("\n")[2].split("|")
    assert bars[:3] == ['"D#m"d4a4', '"C#"gfe2c2c2', '"D#m"d4a4']


def test_repeat_hook_keeps_every_bar_length_and_section_bar_count():
    before = measure_plan(hook_plan())
    after = measure_plan(repeat_hook(hook_plan(), "chorus")[0])
    for old, new in zip(before, after):
        assert old["section"] == new["section"]
        assert old["bars"] == new["bars"]
        assert old["seconds"] == new["seconds"]


def test_repeat_hook_strips_a_tie_that_would_dangle():
    # `d8-` tied into a bar the target does not have in that position would be a
    # different rhythm; the copied bar is re-articulated instead, at the same
    # length (so the section's total time is unchanged).
    abc = (HEADER + '% chorus\nV: Vocal\n"D#m"d8-|"C#"e8|\nV: Ins\nZ2|\n'
           + '% chorus\nV: Vocal\n"B"g4a4|\nV: Ins\nZ|\n')
    out, _ = repeat_hook(abc, "chorus")
    copied = out.split("% chorus")[-1]
    assert "d8-" not in copied
    assert '"D#m"d8|' in copied
    assert measure_plan(out)[-1]["seconds"] == measure_plan(abc)[-1]["seconds"]


def test_repeat_hook_refuses_a_bar_of_a_different_length():
    # The meter is global, so a bar that does not hold a bar's worth of time
    # means the plan is not what we think it is: keep the original bar rather
    # than change the song's timing.
    abc = (HEADER + '% chorus\nV: Vocal\n"D#m"d8|\nV: Ins\nZ|\n'
           + '% chorus\nV: Vocal\n"B"g2|\nV: Ins\nZ|\n')
    out, report = repeat_hook(abc, "chorus")
    assert '"B"g2|' in out                                  # left alone
    assert report[0]["bars_kept_for_length_mismatch"] == 1


def test_repeat_hook_matches_the_target_register_by_octaves():
    # One delivered take had its first chorus an octave below the later ones, so
    # a straight copy would have dropped the final chorus by an octave -- the
    # opposite of where a final chorus goes. The copy is shifted by whole octaves
    # to land in the register the target already sang (an octave, not a few
    # semitones, so the mode and the chords stay in the song's key).
    abc = (HEADER + '% chorus\nV: Vocal\n"D#m"D4A4|\nV: Ins\nZ|\n'
           + '% chorus\nV: Vocal\n"C#"c4g4|\nV: Ins\nZ|\n')
    out, report = repeat_hook(abc, "chorus")
    copied = out.split("% chorus")[-1].split("\n")[2]
    assert copied.startswith('"D#m"d4a4|')       # the hook, an octave up
    assert report[0]["register_octaves"] == 1


def test_repeat_hook_is_a_noop_with_a_single_occurrence():
    only_one = HEADER + '% chorus\nV: Vocal\n"D#m"d8|\nV: Ins\nZ|\n'
    out, report = repeat_hook(only_one, "chorus")
    assert out == only_one and report == []


def test_apply_ops_runs_the_hook_before_the_density_operators():
    # Order matters: the split has to act on the *copied* melody, otherwise the
    # choruses get densified differently and stop sounding like one hook. The
    # later chorus here is shorter, so its bars must be the copied motif's first
    # bars -- both after the same split treatment.
    out, _ = apply_ops(hook_plan(), ["split chorus:2", "hook chorus"])
    first, second = [b.split("\n")[2].split("|") for b in out.split("% chorus")[1:]]
    assert first[:2] == ['"D#m"d2d2a2a2', '"C#"gfe2c2c2']
    assert second[:1] == first[:1]


def test_split_re_articulates_a_held_note():
    out = rewrite("d8|", "split", 2)
    assert out == "d2d2d2d2|"
    assert units_of(out.rstrip("|")) == 8


def test_split_keeps_odd_lengths_integral():
    # Real plans only ever write whole lengths, so a 3-unit note splits 2+1 --
    # never 1.5+1.5 (see format_units).
    assert rewrite("d3e3|", "split", 2) == "d2de2e|"
    assert split_units(3, 2) == [2.0, 1.0]
    assert split_units(5, 2) == [2.0, 2.0, 1.0]
    assert split_units(12, 5) == [4.0, 4.0, 4.0]


def test_split_drops_the_tie_it_replaces():
    # `d8-|d8` is one 16-unit note; re-articulated it is 8 fresh onsets, and no
    # tie may survive (its continuation bar is re-written too).
    out = rewrite("d8-|d8|", "split", 2)
    assert out == "d2d2d2d2|d2d2d2d2|"


def test_format_units_only_writes_lengths_the_plans_use():
    assert format_units(1) == ""
    assert format_units(4) == "4"
    assert format_units(0.5) == "1/2"
    assert format_units(1.5) == "3/2"


def test_chord_symbols_are_never_altered():
    for mode, amount in (("reduce", 2), ("split", 2)):
        out = rewrite('"D#m"d4a4|"C#"g4z3d|', mode, amount)
        assert out.count('"D#m"') == 1 and out.count('"C#"') == 1
        assert "#" in out and "D5#m" not in out and '"Dm"' not in out


def two_section_plan() -> str:
    return (HEADER + "% verse\nV: Vocal\n\"D#m\"d2d2d2ef-|\nV: Ins\nZ|\n"
            "% chorus\nV: Vocal\n\"D#m\"d4a4|\nV: Ins\nZ|\n")


def test_apply_shape_touches_only_matching_sections():
    out, report = apply_shape(two_section_plan(), ["verse:2"], [])
    assert [r["section"] for r in report] == ["verse"]
    assert '"D#m"d2d2d2ef-' not in out           # the verse was rewritten
    assert '"D#m"d4a4|' in out                    # the chorus was not


def test_apply_shape_leaves_the_instrumental_voice_byte_identical():
    ins_lines = ["d4z4|", "z8|"]
    abc = (HEADER + "% verse\nV: Vocal\n\"D#m\"d2d2d2ef-|\nV: Ins\n"
           + ins_lines[0] + "\n% chorus\nV: Vocal\n\"D#m\"d4a4|\nV: Ins\n"
           + ins_lines[1] + "\n")
    out, _ = apply_shape(abc, ["verse:2"], ["chorus:2"])
    assert f"V: Ins\n{ins_lines[0]}\n" in out
    assert f"V: Ins\n{ins_lines[1]}\n" in out


def test_every_bar_still_sums_to_the_same_length_and_density_moves():
    abc = two_section_plan()
    shaped, _ = apply_shape(abc, ["verse:2"], ["chorus:2"])
    before, after = measure_plan(abc), measure_plan(shaped)
    for old, new in zip(before, after):
        assert old["bars"] == new["bars"]
        assert old["seconds"] == new["seconds"]
        # Reducing density drops onsets, and a dropped onset takes its pitch
        # with it, so a range may narrow -- but it may never widen, i.e. no
        # pitch is invented and no section drifts to another register.
        if old["lo_midi"] is not None:
            assert new["lo_midi"] >= old["lo_midi"]
            assert new["hi_midi"] <= old["hi_midi"]
    assert before[0]["onsets"] > after[0]["onsets"]   # verse spread out
    assert before[1]["onsets"] < after[1]["onsets"]   # chorus denser


def test_reduce_bar_reports_no_change_when_there_is_nothing_to_drop():
    items = parse_field("d8|")
    assert reduce_bar(items, 1, False) == items


def test_chorus_operator_does_not_match_pre_chorus():
    # 2026-10-26: substring matching made "hook chorus" copy the chorus into
    # the pre-chorus and "split chorus" densify the launch ramp.  The pre-chorus
    # must be left alone.
    from plan_shape import section_matches

    assert section_matches("% pre-chorus", "chorus") is False
    assert section_matches("% chorus", "chorus") is True
    assert section_matches("% final chorus", "chorus") is True


def test_hook_skips_a_target_much_longer_than_its_source():
    # 2026-10-26: a 2-bar chorus tiled over a 25-bar section is a 12x loop.
    # With a max-tile guard the model's own long-chorus melody is kept.
    target = "|".join(['"C"efga'] * 10) + "|"
    abc = (HEADER
           + "% chorus\nV: Vocal\n\"C\"cdef|\"C\"gabc'|\nV: Ins\nZ2|\n"
           + "% chorus\nV: Vocal\n" + target + "\nV: Ins\nZ10|\n")
    out, report = apply_ops(abc, ["hook chorus"], hook_max_tile=3.5)
    assert any(entry.get("skipped") for entry in report)
    assert target in out
