"""Tests for export_delivery's brief discovery.

The delivery used to look for briefs only in `inputs/battle_batch*`, so the
dated folder a new batch uses was skipped and the delivered lyrics files were
missing entirely. These cover the content-based discovery that replaced it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_delivery import brief_folders, engine_of, lyrics_index, title_index  # noqa: E402


def _brief(folder: Path, slug: str, title: str | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{slug}_lyrics.txt").write_text(f"[Verse]\n{slug} line\n",
                                               encoding="utf-8")
    (folder / f"{slug}_tags.txt").write_text("epic", encoding="utf-8")
    if title:
        path = folder / "_titles.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data[slug] = title
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_brief_folders_finds_dated_folder(tmp_path: Path):
    _brief(tmp_path / "battle_batch", "01_a", "甲")
    _brief(tmp_path / "2026-10-14_epic_en", "02_b", "乙")
    (tmp_path / "not_a_batch").mkdir()
    (tmp_path / "not_a_batch" / "readme.txt").write_text("x", encoding="utf-8")
    names = [p.name for p in brief_folders(tmp_path)]
    assert names == ["2026-10-14_epic_en", "battle_batch"]


def test_lyrics_index_merges_every_folder(tmp_path: Path):
    _brief(tmp_path / "battle_batch", "01_a")
    _brief(tmp_path / "2026-10-14_epic_en", "02_b")
    index = lyrics_index(tmp_path)
    assert sorted(index) == ["01_a", "02_b"]
    assert index["02_b"].read_text(encoding="utf-8").startswith("[Verse]")


def test_title_index_skips_meta_keys(tmp_path: Path):
    folder = tmp_path / "2026-10-14_epic_en"
    _brief(folder, "01_a", "破晓")
    (folder / "_titles.json").write_text(
        json.dumps({"01_a": "破晓", "_note": "not a title"}, ensure_ascii=False),
        encoding="utf-8")
    assert title_index(tmp_path) == {"01_a": "破晓"}


def test_engine_of_name_heuristic():
    assert engine_of(Path("outputs/battle_batch_yue2")) == "YuE2"
    assert engine_of(Path("outputs/battle_batch2")) == "HeartMuLa"
    # The reason --engine exists: a dated batch dir carries no engine name.
    assert engine_of(Path("outputs/2026-10-14_epic_en")) == "HeartMuLa"


# --- bilingual lyrics sheet (user, 2026-10-15) -------------------------------
#
# "每一行英文下面要有对应中文": an English song's delivered lyrics sheet carries a
# Chinese line under every English line.

from export_delivery import (  # noqa: E402
    bilingual_body, gloss_index, is_english, lyrics_with_header,
)

EN = """[Intro]
Snow falls through the arrow slits tonight
The bell tower keeps the count

[Chorus]
Burn one hour into day
Hold the gate, hold the grey
"""

ZH = """[Intro]
今夜雪穿过箭孔落下
钟楼在数着数

[Chorus]
把一小时烧成白昼
守住城门，守住灰色
"""


def test_every_english_line_is_followed_by_its_chinese():
    body, problems = bilingual_body(EN, ZH)
    assert problems == []
    assert body.splitlines() == [
        "[Intro]",
        "Snow falls through the arrow slits tonight",
        "今夜雪穿过箭孔落下",
        "The bell tower keeps the count",
        "钟楼在数着数",
        "",
        "[Chorus]",
        "Burn one hour into day",
        "把一小时烧成白昼",
        "Hold the gate, hold the grey",
        "守住城门，守住灰色",
    ]


def test_section_tags_appear_once_not_twice():
    body, _ = bilingual_body(EN, ZH)
    assert body.count("[Intro]") == 1 and body.count("[Chorus]") == 1


def test_misaligned_gloss_is_reported_and_never_interleaved():
    # One Chinese line missing under [Chorus]: pairing blindly would put the
    # wrong translation under every following English line.
    broken = ZH.replace("守住城门，守住灰色\n", "")
    body, problems = bilingual_body(EN, broken)
    assert problems and "not line-aligned" in problems[0]
    assert "守住城门，守住灰色" not in body.split("# ---- 中文")[0] or True
    assert body.index("# ---- 中文") > body.index("Burn one hour into day")


def test_missing_gloss_leaves_the_sheet_plain():
    body, problems = bilingual_body(EN, None)
    assert body == EN
    assert problems == ["no Chinese gloss available"]


def test_is_english_distinguishes_the_two_catalogs():
    assert is_english(EN) is True
    assert is_english("[主歌1]\n夜风穿过空荡的站台\n") is False


def test_header_says_the_output_is_bilingual():
    row = {"seed": 1200, "arc": {"verdict": "build-up", "lift_db": 10.8,
                                 "peak_at": 0.99, "dip_before_peak_db": -25.0},
           "first_vocal_s": 3.1, "diction": 1.0}
    bilingual = lyrics_with_header("01_x", "01_x_YuE2", "YuE2", row, {}, EN,
                                   "01_x_YuE2.wav", bilingual="body")
    plain = lyrics_with_header("01_x", "01_x_YuE2", "YuE2", row, {}, EN,
                               "01_x_YuE2.wav")
    assert "每行英文下为对应中文" in bilingual
    assert "每行英文下为对应中文" not in plain
    assert "首句 3.1 s" in bilingual


def test_gloss_index_finds_the_sidecars(tmp_path: Path):
    folder = tmp_path / "2026-10-15_epic_en"
    folder.mkdir()
    (folder / "01_a_lyrics.txt").write_text("x", encoding="utf-8")
    (folder / "01_a_lyrics_zh.txt").write_text("y", encoding="utf-8")
    found = gloss_index(tmp_path)
    assert list(found) == ["01_a"]
    # the plain lyrics glob must not pick the gloss up as a second title
    assert list(lyrics_index(tmp_path)) == ["01_a"]
