"""Tests for the YuE2 plan editor: shortening the instrumental intro.

The fixture mirrors a real plan saved by the 2026-10-14 run (see
`.../01_last_dawn_city_c1_artifacts/score.abc`): an 18-bar intro of silent vocal
bars, then the sung sections. The measured first-vocal onset of that take was
30.0 s, and 18 bars x 1.644 s = 29.6 s, which is the whole complaint.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yue2_plan_edit import (  # noqa: E402
    chunk_spans, section_bars, seconds_per_bar, split_plan, trim_intro,
)

PLAN = """X:1
T:
M:2/4
L:1/16
Q:1/4=73
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:D#m
% intro
V: Vocal
"D#m"z8|"D#m"z8|"D#m"z8|"D#m"z8|
V: Ins
Z|z4D2A2|G6AG|F4E4|
V: Vocal
"D#m"z8|"D#m"z8|"D#m"z8|"D#m"z8|
V: Ins
D8-|D4D2A2|G6AG|F4E4|
V: Vocal
"D#m"z8|"D#m"z8|"Bmaj7"z8|"Bmaj7"z8|
V: Ins
A8-|A4d2a2|g6ag|f4e4|
V: Vocal
"D#m"z8|"D#m"z8|"C#"z8|"C#"z8|
V: Ins
d8-|d4d2a2|g6ag|f4e4|
V: Vocal
"D#m"z8|"D#m"z8|
V: Ins
d8-|d8|
% verse
V: Vocal
"D#m"d2d2d2ef-|"D#m"f2e2d2c2|
V: Ins
Z2|
% outro
V: Vocal
"D#m"d2z6|
V: Ins
Z|
"""


def test_header_and_sections_are_split():
    header, sections = split_plan(PLAN)
    assert header[0] == "X:1" and header[-1] == "K:D#m"
    assert [name for name, _ in sections] == ["% intro", "% verse", "% outro"]


def test_intro_bars_count_the_vocal_voice_only():
    # The `V: Ins` line sounds under the same bars; counting both would report
    # 36 bars for this intro and trim half of it.
    _, sections = split_plan(PLAN)
    intro = dict(sections)["% intro"]
    assert section_bars(intro) == 18
    assert [bars for _, _, bars in chunk_spans(intro)] == [4, 4, 4, 4, 2]


def test_seconds_per_bar_comes_from_meter_and_tempo():
    # 2/4 at a quarter = 73 bpm -> 2 x 60/73 = 1.644 s
    assert abs(seconds_per_bar(PLAN) - 1.644) < 0.01


def test_trim_is_bar_precise():
    for keep in (1, 2, 4, 8):
        _, info = trim_intro(PLAN, keep_bars=keep)
        assert info["intro_bars_after"] == keep, keep
        assert info["dropped_bars"] == 18 - keep
        assert info["changed"] is True


def test_trim_reports_the_seconds_it_saved():
    # 16 bars x 1.644 s = 26.3 s: enough to turn a 207 s take into a 181 s one
    # that starts singing at ~3 s instead of 30 s.
    _, info = trim_intro(PLAN, keep_bars=2)
    assert abs(info["dropped_seconds"] - 26.3) < 0.2


def test_trim_keeps_every_other_section_byte_identical():
    new, _ = trim_intro(PLAN, keep_bars=2)
    assert new.endswith(PLAN[PLAN.index("% verse"):])


def test_trim_keeps_vocal_and_instrumental_in_phase():
    new, _ = trim_intro(PLAN, keep_bars=2)
    intro = new[new.index("% intro"):new.index("% verse")]
    vocal_bars = [ln for ln in intro.splitlines() if ln.startswith('"')]
    assert vocal_bars and all(ln.count("|") == 2 for ln in vocal_bars)
    ins = intro.splitlines()[intro.splitlines().index("V: Ins") + 1]
    assert ins.count("|") == 2


def test_short_intro_is_left_alone():
    _, info = trim_intro(PLAN, keep_bars=32)
    assert info["changed"] is False
    assert info["dropped_bars"] == 0


def test_keep_bars_zero_still_leaves_a_usable_intro():
    new, info = trim_intro(PLAN, keep_bars=0)
    assert info["intro_bars_after"] == 1
    assert "% verse" in new


def test_plan_without_section_markers_is_returned_unchanged():
    abc = "X:1\nM:4/4\nK:Cm\nV: Vocal\nc4|c4|\n"
    new, info = trim_intro(abc)
    assert new == abc
    assert info["changed"] is False


def test_voices_that_disagree_on_bar_count_fall_back_to_whole_chunks():
    # A chunk whose Ins line is shorter than the slice we need must not be
    # half-cut: that would leave the two voices out of phase.
    broken = PLAN.replace('V: Ins\nZ|z4D2A2|G6AG|F4E4|', 'V: Ins\nZ|z4D2A2|', 1)
    _, info = trim_intro(broken, keep_bars=2)
    assert info["changed"] is True
    assert info["intro_bars_after"] <= 4      # 18 -> 6 whole bars, never 2
    assert info["intro_bars_after"] > 0


# --- final-chorus pitch lift (2026-10-15) ------------------------------------
#
# "高的时候不够高": in every plan inspected the final chorus repeats the first
# chorus's chords and melody verbatim, so the step-up modulation the reference
# songs use (Starfall Dm -> D) has to be written into the plan.

from yue2_plan_edit import (  # noqa: E402
    _note_midi, _midi_to_abc_note, key_accidentals, transpose_section,
)


def test_key_signature_of_d_sharp_minor_is_six_sharps():
    assert key_accidentals("D#m") == {"F": 1, "C": 1, "G": 1, "D": 1, "A": 1, "E": 1}
    assert key_accidentals("C") == {}
    assert key_accidentals("F") == {"B": -1}


def test_abc_octave_conventions():
    key: dict[str, int] = {}
    assert _note_midi("C", "", "", key) == 60      # middle C
    assert _note_midi("c", "", "", key) == 72      # an octave up
    assert _note_midi("B", "", "", key) == 71
    assert _note_midi("b", "", "", key) == 83
    assert _note_midi("c", "", "'", key) == 84
    assert _note_midi("C", "", ",", key) == 48
    assert _midi_to_abc_note(60, key, 2) == "C"
    assert _midi_to_abc_note(72, key, 2) == "c"
    assert _midi_to_abc_note(73, key, 2) == "^c"


def test_key_signature_is_applied_to_unmarked_notes():
    # In D#m an unmarked `d` is D#5. Two semitones up is F natural, which the key
    # signature does not imply, so the output has to spell it out.
    key = key_accidentals("D#m")
    assert _note_midi("d", "", "", key) == 75          # D#5
    assert _midi_to_abc_note(77, key, 2) == "=f"       # F natural, written out
    assert _midi_to_abc_note(80, key, 2) == "g"        # G#5 still in the key


def test_original_spelling_is_kept_when_the_pitch_allows_it():
    # E sharp in D#m stays `e` (the key signature already makes it sharp) rather
    # than being respelled `=f`, which is why transposing by zero is a no-op.
    key = key_accidentals("D#m")
    assert _note_midi("e", "", "", key) == 77          # E#5
    assert _midi_to_abc_note(77, key, 0, prefer_letter="e") == "e"
    assert _midi_to_abc_note(77, key, 0) == "=f"       # without the preference
    # a chord root is not governed by the key signature, so it stays explicit
    assert _midi_to_abc_note(63, key, 0, prefer_letter="d",
                             always_explicit=True) == "^D"


SONG = """X:1
T:
M:2/4
L:1/16
Q:1/4=73
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:D#m
% intro
V: Vocal
"D#m"z8|"D#m"z8|
V: Ins
d8-|d8|
% verse
V: Vocal
"D#m"d2d2d2ef-|"D#m"f2e2d2c2|
V: Ins
Z2|
% chorus
V: Vocal
"D#m"d4a4|"C#"gfe2c2c2|
V: Ins
Z2|
% verse
V: Vocal
"D#m"a3za2a2|"C#"gfe2c2c2|
V: Ins
Z2|
% chorus
V: Vocal
"D#m"d4a4|"C#"gfe2c2c2|
V: Ins
Z2|
% outro
V: Vocal
"D#m"d2z6|
V: Ins
Z|
"""


def test_transpose_section_moves_only_the_last_chorus():
    new, info = transpose_section(SONG, section="chorus", semitones=2)
    assert info["changed"] is True
    assert info["section"] == "% chorus"
    # intro, verses and outro are byte-identical
    for marker in ("% intro", "% verse", "% outro"):
        assert SONG[SONG.index(marker):SONG.index(marker) + 60] in new
    first_chorus = SONG[SONG.index("% chorus"):SONG.index("% verse", SONG.index("% chorus"))]
    assert first_chorus in new          # the FIRST chorus is untouched


def test_transposed_final_chorus_is_higher_and_coherent():
    key = key_accidentals("D#m")
    new, _ = transpose_section(SONG, section="chorus", semitones=2)
    final = new[new.rindex("% chorus"):new.index("% outro")]
    # original opens "D#m"d4a4 -> +2 semitones = Fm, both notes out of the key
    assert '"Fm"' in final
    assert "=f4" in final          # D#5 -> F natural
    assert "=c'4" in final         # A#5 -> C natural, an octave marker up
    # and the pitch really is a step up: D#5 -> F5
    assert _note_midi("d", "", "", key) + 2 == _note_midi("f", "=", "", key)


def test_missing_section_is_reported_not_crashed():
    new, info = transpose_section(SONG, section="pre-chorus", semitones=2)
    assert new == SONG
    assert info["changed"] is False


def test_transpose_zero_semitones_changes_nothing():
    new, _ = transpose_section(SONG, section="chorus", semitones=0)
    assert new == SONG
