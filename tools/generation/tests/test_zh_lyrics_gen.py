"""Tests for the Chinese brief generator's local gate.

These do not call DeepSeek: they pin what the script accepts/rejects once a
response arrives, so a model regression cannot silently put an English line, a
wordless hook, or a non-open-vowel hook into the render queue.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_zh_lyric_qa import GOOD  # noqa: E402
from zh_lyrics_gen import build_tags, validate_brief  # noqa: E402

SEED = {"mode": "narrative", "bpm": 96, "key": "C major"}


def brief(**overrides):
    data = {
        "title": "末班车以后",
        "hook": "末班车以后 你还在不在",
        "lyrics": GOOD,
        "style_extra": "warm piano, soft brushed drums, intimate room",
        "notes": "便利店门口的两个人，谁都没有先开口。",
    }
    data.update(overrides)
    return data


def test_good_brief_passes():
    assert validate_brief(brief(), SEED) == []


def test_english_line_is_rejected():
    broken = brief(lyrics=GOOD.replace("雨停在便利店的玻璃门", "I love you"))
    problems = validate_brief(broken, SEED)
    assert any("英文" in p for p in problems)


def test_wordless_line_is_rejected():
    broken = brief(lyrics=GOOD.replace("我把那句喜欢 留在找零里猜", "呀 咿 呀 咿 呀"))
    problems = validate_brief(broken, SEED)
    assert any("拟声" in p or "哼唱" in p for p in problems)


def test_missing_hook_is_rejected():
    broken = brief(hook="不存在的 hook")
    problems = validate_brief(broken, SEED)
    assert any("hook" in p for p in problems)


def test_hook_must_end_on_an_open_vowel():
    # Replace the hook everywhere with a closed -i ending.  The lyric still has
    # a repeated hook, so only the open-vowel rule should catch it.
    closed = GOOD.replace("末班车以后 你还在不在", "末班车以后 你别走开")  # 开 is open
    assert validate_brief(brief(lyrics=closed, hook="末班车以后 你别走开"), SEED) == []
    closed = GOOD.replace("末班车以后 你还在不在", "末班车以后 你别离去")
    problems = validate_brief(brief(lyrics=closed, hook="末班车以后 你别离去"), SEED)
    assert any("开口音" in p for p in problems)


def test_chinese_style_extra_is_rejected():
    problems = validate_brief(brief(style_extra="温暖的钢琴，柔软鼓刷"), SEED)
    assert any("英文" in p for p in problems)


def test_tags_are_female_only_and_forbid_wordless_vocalise():
    tags = build_tags(SEED, "warm piano, soft drums")
    assert "female vocal" in tags
    assert re.search(r"(?<!fe)male vocal", tags) is None
    assert "no onomatopoeia" in tags
    assert "no synthetic humming" in tags


def test_epic_brief_requires_metaphor_and_philosophy():
    seed = {"mode": "epic", "bpm": 84, "key": "E minor"}
    data = brief(metaphor="", philosophy="")
    problems = validate_brief(data, seed)
    assert any("metaphor" in p for p in problems)
    assert any("philosophy" in p for p in problems)
