"""Tests for the symbolic melody audit (no GPU, no audio).

The numbers are read from the ABC plan, so these fixtures are tiny hand-written
plans where the expected answer is obvious by inspection.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from melody_qa import audit  # noqa: E402

HEADER = "X:1\nT:\nM:4/4\nL:1/8\nQ:1/4=100\nV: Vocal\nV: Ins\nK:C\n"


def plan(chorus: str, final: str) -> str:
    return (
        HEADER
        + "% verse\nV: Vocal\n\"C\"cdefgabc'|\"C\"cbagfedc|\nV: Ins\nZ2|\n"
        + "% chorus\nV: Vocal\n" + chorus + "\nV: Ins\nZ4|\n"
        + "% final chorus\nV: Vocal\n" + final + "\nV: Ins\nZ4|\n"
    )


def test_density_and_range_are_read_from_the_plan():
    # Verse: 16 eighths over 2 bars.  Chorus: 8 half-notes over 4 bars, so the
    # chorus is sparser -- the exact failure the 2026-10-16 measurements found.
    report = audit(plan(
        '"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|',
        '"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|',
    ), target="diagnostic")
    assert report.verse_density > report.chorus_density
    assert report.density_ratio < 1.4
    assert report.climax_midi == 84
    assert report.overall_range == 12


def test_same_pitch_classes_across_choruses_are_one_hook():
    same = '"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|'
    report = audit(plan(same, same), target="diagnostic")
    assert report.hook_similarity is not None
    assert report.hook_similarity == 1.0


def test_transposed_final_chorus_is_still_the_same_hook():
    # e'/a is the chorus c'/g transposed up a major third: the final chorus was
    # lifted, not rewritten. Hook similarity must be transposition-invariant.
    chorus = '"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|'
    final = '"C"e\'2e\'2a2a2|"C"e\'2e\'2a2a2|"C"e\'2e\'2a2a2|"C"e\'2e\'2a2a2|'
    report = audit(plan(chorus, final), target="diagnostic")
    assert report.hook_similarity == 1.0


def test_different_hook_shape_lowers_similarity():
    # A major third (c'/e) is not a transposition of the perfect fifth (c'/g);
    # this is a genuinely different hook, not a lifted one.
    chorus = '"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|"C"c\'2c\'2g2g2|'
    final = '"C"c\'2c\'2e2e2|"C"c\'2c\'2e2e2|"C"c\'2c\'2e2e2|"C"c\'2c\'2e2e2|'
    report = audit(plan(chorus, final), target="diagnostic")
    assert report.hook_similarity is not None
    assert report.hook_similarity < 0.95


def test_zh_pop_target_rejects_flat_dynamics_and_early_climax():
    # Same melody in verse and chorus: no density contrast, no register lift,
    # and the highest note appears in the verse -> three failures at once.
    flat = ('"C"cdefgabc\'|"C"cbagfedc|"C"cdefgabc\'|"C"cbagfedc|')
    report = audit(plan(flat, flat), target="zh-pop")
    assert not report.passes
    assert any("密度" in issue for issue in report.issues)
    assert any("音区" in issue for issue in report.issues)


def test_missing_chorus_is_critical():
    abc = HEADER + "% verse\nV: Vocal\n\"C\"cdefgabc'|\nV: Ins\nZ1|\n"
    report = audit(abc, target="zh-pop")
    assert not report.passes
    assert any("缺少主歌或副歌" in issue for issue in report.issues)
