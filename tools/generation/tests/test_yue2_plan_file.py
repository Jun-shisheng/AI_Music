"""Tests for rendering from a pre-shaped plan (`--plan-file`).

2026-10-16: this is the seam that lets a plan edited by `plan_shape.py` be
rendered, which is how the density experiment (docs/arrangement_spec.md §1.7)
isolates "did the words get denser" from "did the seed change". A plan file that
lost its structure would still render -- as an unrelated song -- so it is
validated before it reaches the pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yue2_batch_battle import plan_from_file  # noqa: E402

GOOD = ('X:1\nM:2/4\nL:1/16\nQ:1/4=76\nV: Vocal\nV: Ins\nK:D#m\n'
        '% verse\nV: Vocal\n"D#m"d8|\nV: Ins\nZ|\n')


def test_reads_a_shaped_plan(tmp_path: Path):
    path = tmp_path / "score.abc"
    path.write_text(GOOD, encoding="utf-8")
    assert '"D#m"d8|' in plan_from_file(path)


def test_tolerates_a_utf8_bom(tmp_path: Path):
    # plan_shape.py writes utf-8; the renderer's reader must not choke on a BOM.
    path = tmp_path / "score.abc"
    path.write_text(GOOD, encoding="utf-8-sig")
    assert plan_from_file(path).startswith("X:1")


def test_rejects_a_file_without_section_markers(tmp_path: Path):
    path = tmp_path / "score.abc"
    path.write_text("X:1\nV: Vocal\nd8|\n", encoding="utf-8")
    with pytest.raises(ValueError, match="section"):
        plan_from_file(path)


def test_rejects_a_plan_with_no_vocal_voice(tmp_path: Path):
    path = tmp_path / "score.abc"
    path.write_text("X:1\n% verse\nV: Ins\nZ|\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Vocal"):
        plan_from_file(path)
