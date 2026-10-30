"""Tests for the Mandarin lyric gate.

The gate exists to stop a bad brief before a 6-20 minute GPU render.  These
tests pin the behaviours that the 2026-10-26 brief actually depends on:
a concrete, hook-first lyric passes; a cliche/wordless/no-hook lyric fails;
the three modes (narrative / atmospheric / guofeng) enforce different budgets.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zh_lyric_qa import audit  # noqa: E402

GOOD = """[Intro]
雨停在便利店的玻璃门

[Verse 1]
你把关东煮推到我面前
塑料叉子还冒着一点烟
末班车从街口拐了个弯
我数着找零里的硬币声

[Pre-Chorus 1]
你没问 我也没讲
把纸盒轻轻往你那边放

[Chorus]
末班车以后 你还在不在
玻璃门上的雨 擦了几遍才白
末班车以后 你还没走开
我把那句喜欢 留在找零里猜

[Verse 2]
白灯照着你湿掉的袖口
你说今晚风有点凉透
我把外套搭在椅背后面
假装没看见你握紧的手

[Pre-Chorus 2]
你没问 我也没讲
把纸盒轻轻往你那边放

[Chorus 2]
末班车以后 你还在不在
路灯把影子拉得很窄
你把剩下的汤递过来
说凉了就不好喝了

[Bridge]
其实两张车票 早就买好
一直放在我外套的内侧口袋

[Final Chorus]
末班车以后 你还在不在
这一次我先把手伸出来
雨已经停了 街口很白
你不用再假装看站牌

[Outro]
关东煮的纸盒 还留在门外
"""


def test_concrete_hook_first_lyric_passes():
    report = audit(GOOD, title="末班车以后", mode="narrative")
    assert report.passes, report.issues
    assert report.hook == "末班车以后你还在不在"
    assert report.hook_count == 3
    assert len(report.concrete_hits) >= 4
    assert report.rhyme_by_section["chorus"] >= 0.5


def test_no_repeated_hook_fails():
    broken = GOOD.replace("[Chorus 2]", "[Chorus 2]").replace(
        "末班车以后 你还在不在\n路灯把影子拉得很窄",
        "末班车已经开走了\n路灯把影子拉得很窄").replace(
        "末班车以后 你还在不在\n这一次我先把手伸出来",
        "站牌上的字看不清了\n这一次我先把手伸出来")
    report = audit(broken, mode="narrative")
    assert not report.passes
    assert any("hook" in issue for issue in report.issues)


def test_ai_copywriter_cliche_fails():
    broken = GOOD.replace("我把那句喜欢 留在找零里猜",
                          "我奔赴山海 未来可期")
    report = audit(broken, mode="narrative")
    assert not report.passes
    assert report.cliche_hits


def test_wordless_line_is_rejected_even_in_atmospheric_mode():
    broken = GOOD.replace("我把那句喜欢 留在找零里猜", "呀 咿 呀 咿 呀")
    report = audit(broken, mode="atmospheric")
    assert not report.passes
    assert any("拟声" in issue or "哼唱" in issue for issue in report.issues)


def test_atmospheric_mode_allows_a_shorter_lyric():
    # 23 sung lines, same tag order, real-word hook; the narrative gate would
    # reject this on line count, the atmospheric gate should not.
    short = """[Intro]
电梯停在没有人按的楼层

[Verse 1]
走廊的声控灯亮了一下
防盗门后面没有脚步声
手机录音的红点还亮着

[Pre-Chorus 1]
我没有发出去

[Chorus]
凌晨三点 你还留着钥匙
冰箱的嗡鸣替我回答
凌晨三点 你还留着钥匙
猫眼里没有人 只有走廊的白

[Verse 2]
厨房的水杯还剩一半
你走那天忘了关上窗
风把窗帘吹到我脸上

[Pre-Chorus 2]
我没有关掉它

[Chorus 2]
凌晨三点 你还留着钥匙
电梯又停在这一层
门开的时候 我听见自己
把那个名字咽了回去

[Bridge]
那段语音其实已经录好

[Final Chorus]
凌晨三点 你还留着钥匙
这一次我没有假装睡着
走廊很白 电梯很慢
我把门打开了一条缝

[Outro]
灯又灭了
"""
    report = audit(short, mode="atmospheric")
    assert report.line_count == 23
    assert report.hook == "凌晨三点你还留着钥匙"
    # It may fail on concreteness/tone, but it must not fail on the narrative
    # line budget or on the section tags.
    assert not any("总句数" in issue for issue in report.issues)
    assert not report.section_errors


def test_guofeng_mode_accepts_a_slogan_hook():
    lyric = """[Intro]
雪落在城门外的旧甲上

[Verse 1]
我把名字写在剑鞘背面
火把照着马蹄印往北
酒碗里还有半口没喝完
城墙上的霜比刀还白

[Pre-Chorus 1]
不等援军
只等门开

[Chorus]
我本该自己开天
这一剑由我自己递出
雪落在旧甲的裂缝里
不求仙也不求神

[Verse 2]
旧甲上的裂缝藏着风
有人跪着求一个前程
我把最后一口酒倒进雪
听见城门在里面松动

[Pre-Chorus 2]
不等天亮
只等门开

[Chorus 2]
我本该自己开天
这一剑由我自己递出
让雪替我记住来路
不求仙也不求神

[Bridge]
他们说的天命
不过是一道没关紧的门

[Final Chorus]
我本该自己开天
这一次我不再等谁开门
剑锋朝上 雪落无声
我自己的路 我自己认

[Outro]
城门开了
"""
    report = audit(lyric, mode="guofeng")
    assert report.hook == "我本该自己开天"
    assert report.hook_count == 3
    assert not report.section_errors
