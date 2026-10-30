"""Tests for publish_delivery's pure parts (sheet rewrite, commit message, naming).

The git and export steps are I/O against the real repo and are exercised by
running the thing; what is covered here is the text handling that made the
2026-10-14 hand-published delivery correct -- and whose absence would quietly
produce a sheet naming files that are not in the folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from publish_delivery import commit_message, delivery_sheet  # noqa: E402

SHEET = """# 战斗曲试听清单 (battle batch)

- 候选规则：每首多个候选
- 交付文件：`selected/<slug>.wav`（线性响度归一化）；原件：`selected/<slug>_raw.wav`（零处理，保留）

| 引擎 | 曲目 | 文件 |
|---|---|---|
| YuE2 | 破晓前一小时 `01_last_dawn_city` | `01_last_dawn_city.wav` |
"""


def test_sheet_file_column_points_at_delivered_names():
    out = delivery_sheet(SHEET, "epic_en", "YuE2", 9)
    assert "`01_last_dawn_city_YuE2.wav`" in out
    assert "`01_last_dawn_city.wav`" not in out


def test_sheet_header_names_the_delivery_and_count():
    out = delivery_sheet(SHEET, "epic_en", "YuE2", 9)
    assert out.splitlines()[0] == "# 战斗曲试听清单 — epic_en（YuE2，9 首）"


def test_sheet_explains_the_folder_layout():
    out = delivery_sheet(SHEET, "epic_en", "YuE2", 9)
    assert "`<slug>_YuE2.wav`" in out
    assert "selected/<slug>_raw.wav" in out
    assert "交付文件：`selected/<slug>.wav`" not in out


def test_sheet_leaves_two_digit_but_non_slug_backticks_alone():
    out = delivery_sheet("- see `01_note` and `manifest.json`\n", "x", "YuE2", 1)
    assert "`01_note`" in out and "`manifest.json`" in out


def test_commit_message_lists_tracks():
    msg = commit_message("2026-10-14_epic_en", 3, "YuE2", ["10_a", "11_b", "12_c"])
    assert msg.splitlines()[0] == "2026-10-14_epic_en: 3 tracks (YuE2)"
    assert "- 11_b" in msg
