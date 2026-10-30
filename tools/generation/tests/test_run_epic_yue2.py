"""Tests for run_epic_yue2: the length gate, the brief discovery and the state file.

The user's bound is explicit -- 2:00 to 3:30 is right, 1:30 is too short -- and
YuE2 samples its own length, so this gate is what enforces it. The subprocess
loop itself is not exercised here (it needs the 7 GB model), but every decision
it branches on is.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_epic_yue2 import needs_rerun, select_slugs, title_slugs, write_status  # noqa: E402

MIN_S, MAX_S = 120.0, 210.0


def test_short_take_is_rejected():
    assert "too short" in needs_rerun(95.0, MIN_S, MAX_S)


def test_long_take_is_rejected():
    assert "too long" in needs_rerun(241.0, MIN_S, MAX_S)


def test_window_is_inclusive():
    assert needs_rerun(120.0, MIN_S, MAX_S) is None
    assert needs_rerun(210.0, MIN_S, MAX_S) is None
    assert needs_rerun(191.8, MIN_S, MAX_S) is None


def test_boundary_neighbours_are_flagged():
    assert needs_rerun(119.9, MIN_S, MAX_S) is not None
    assert needs_rerun(210.1, MIN_S, MAX_S) is not None


def test_title_slugs_pairs_lyrics_with_tags(tmp_path: Path):
    (tmp_path / "02_b_lyrics.txt").write_text("x", encoding="utf-8")
    (tmp_path / "02_b_tags.txt").write_text("x", encoding="utf-8")
    (tmp_path / "01_a_lyrics.txt").write_text("x", encoding="utf-8")
    (tmp_path / "01_a_tags.txt").write_text("x", encoding="utf-8")
    # orphan lyrics with no tags file: not a brief
    (tmp_path / "03_c_lyrics.txt").write_text("x", encoding="utf-8")
    assert title_slugs(tmp_path) == ["01_a", "02_b"]


def test_status_file_is_plain_text(tmp_path: Path):
    path = tmp_path / "RUN_STATUS.txt"
    write_status(path, current="05_spearstorm", done="4/12", elapsed_min="88.5")
    body = path.read_text(encoding="utf-8")
    assert body == "current: 05_spearstorm\ndone: 4/12\nelapsed_min: 88.5\n"


def test_slug_filter_restricts_the_run():
    slugs = ["01_a", "10_b", "11_c", "12_d"]
    assert select_slugs(slugs, ["10_", "11_", "12_"]) == ["10_b", "11_c", "12_d"]
    assert select_slugs(slugs, None) == slugs
    assert select_slugs(slugs, []) == slugs


def test_slug_filter_matching_nothing_is_empty_not_everything():
    assert select_slugs(["01_a"], ["99_"]) == []


# --- vocal-onset gate (2026-10-15) -------------------------------------------
#
# The user's "前奏太长" rule: the first sung word within a few seconds, or an
# intro of at most 10-20 s. The 2026-10-14 delivery ran 15.5-32.3 s.

from run_epic_yue2 import onset_reason, read_first_vocal  # noqa: E402


def test_late_vocal_entry_is_rejected():
    assert "vocal starts at 30s" in onset_reason(30.0, 20.0)


def test_prompt_vocal_entry_passes():
    assert onset_reason(3.0, 20.0) is None
    assert onset_reason(20.0, 20.0) is None


def test_missing_onset_is_not_a_failure():
    # No ASR venv -> the diction check is skipped, and the gate must not fail
    # an otherwise good title on a metric nobody measured.
    assert onset_reason(None, 20.0) is None
    assert onset_reason(-1.0, 20.0) is None


def test_read_first_vocal_reads_the_batch_index(tmp_path: Path):
    import json
    (tmp_path / "index.json").write_text(json.dumps([
        {"title": "01_a", "first_vocal_s": 3.4},
        {"title": "02_b", "first_vocal_s": None},
    ]), encoding="utf-8")
    assert read_first_vocal(tmp_path, "01_a") == 3.4
    assert read_first_vocal(tmp_path, "02_b") is None
    assert read_first_vocal(tmp_path, "03_c") is None


def test_read_first_vocal_without_index_is_none(tmp_path: Path):
    assert read_first_vocal(tmp_path, "01_a") is None
