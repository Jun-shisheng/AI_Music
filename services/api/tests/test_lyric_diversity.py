from app.services.lyric_diversity import analyze_lyrics, assert_lyric_diversity


GOOD_LYRICS = """[主歌1]
夜风穿过空荡的站台
我把车票折成纸船
远处灯火像未寄的信
口袋里还剩半截夏天

[副歌1]
如果你听见晚班车开过
请把想念放在窗沿
我学会把眼泪说成天气
却说不完对你的挂念

[主歌2]
后来城市换了地图
旧咖啡馆换成便利店
我在人群里练习微笑
却在雨声里想起你的脸

[副歌2]
如果你听见晚班车开过
请把想念放在窗沿
这一次我把故事写短一点
只留下你走过的那条街

[桥段]
世界很大我们很小
错过也是一种答案
"""


FATIGUE_LYRICS = """[主歌1]
我想你我想你我想你
每天都在想着你
我想你我想你我想你
每天都在想着你

[副歌]
爱你爱你爱你爱你
爱你爱你爱你爱你
爱你爱你爱你爱你
爱你爱你爱你爱你

[主歌2]
我想你我想你我想你
每天都在想着你
我想你我想你我想你
每天都在想着你
"""


class TestLyricDiversity:
    def test_good_lyrics_pass(self):
        report = analyze_lyrics(GOOD_LYRICS)
        assert report.passes, report.issues
        assert report.unique_ratio >= 0.7

    def test_fatigue_lyrics_fail(self):
        report = analyze_lyrics(FATIGUE_LYRICS)
        assert not report.passes
        assert report.issues

    def test_assert_raises_on_fatigue(self):
        try:
            assert_lyric_diversity(FATIGUE_LYRICS)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "Lyric diversity" in str(exc)


# --- strict repetition audit (2026-10-15) -----------------------------------
#
# The user's battle-song rule: repetition is allowed but never in large blocks;
# the opening, the climax (final chorus) and the ending must be brand-new text;
# in the middle a phrase may repeat, but twice is the ceiling.

from app.services.lyric_diversity import audit_repetition  # noqa: E402


FRESH_ARC = """[Intro]
Snow falls through the arrow slits tonight
The bell tower keeps the count

[Verse 1]
The signal fire is all we have to see
The arrows come like sleet against the stone

[Chorus]
Burn one hour into day
Hold the gate, hold the grey
Someone is singing in the dark below
Snow keeps falling, I keep watch

[Verse 2]
They breached the lower yard an hour ago
Melted snow runs red between the cracks

[Chorus 2]
Burn one hour into day
Hold the gate, hold the grey
The horn has stopped, the yard is still
I have not slept since yesterday's bell

[Bridge]
The tower bell is cracked, it will not ring
So I make the sound it cannot make

[Final Chorus]
East is pale and I am still standing
Every name I kept is written on the wall
The gate is mine, the grey is mine
Dawn is coming up like a breaking wave

[Outro]
Nobody left to count but me
"""


class TestRepetitionAudit:
    def test_fresh_lyrics_pass(self):
        report = audit_repetition(FRESH_ARC)
        assert report.passes, report.issues
        assert report.max_repeat == 2      # the two chorus lines, no more

    def test_third_repeat_of_a_line_is_rejected(self):
        lyrics = FRESH_ARC.replace(
            "[Final Chorus]",
            "[Final Chorus]\nBurn one hour into day")
        report = audit_repetition(lyrics)
        assert not report.passes
        assert "重复超过 2 次" in report.summary()

    def test_recycled_chorus_as_final_chorus_is_rejected(self):
        lines = FRESH_ARC.splitlines()
        start = lines.index("[Final Chorus]")
        out = lines.index("[Outro]")
        recycled = lines[:start] + [
            "[Final Chorus]",
            "Burn one hour into day",
            "Hold the gate, hold the grey",
            "Someone is singing in the dark below",
            "Snow keeps falling, I keep watch",
        ] + lines[out:]
        report = audit_repetition("\n".join(recycled))
        assert not report.passes
        assert report.stale_sections == ["final chorus"]

    def test_outro_reusing_a_verse_line_is_rejected(self):
        lyrics = FRESH_ARC.replace(
            "[Outro]\nNobody left to count but me",
            "[Outro]\nThe signal fire is all we have to see")
        report = audit_repetition(lyrics)
        assert not report.passes
        assert "outro" in report.stale_sections

    def test_ordered_sections_keeps_both_choruses(self):
        from app.services.lyric_diversity import ordered_sections
        labels = [label for label, _ in ordered_sections(FRESH_ARC)]
        assert labels == ["intro", "verse 1", "chorus", "verse 2", "chorus 2",
                          "bridge", "final chorus", "outro"]
