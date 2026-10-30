from app.services.music_text import detect_vocal_language, prepare_generation_lyrics


def test_prepare_generation_lyrics_removes_intro_words_and_wordless_fillers():
    lyrics = """[Intro]
月色落在城墙边
风把灯影吹得很远

[Verse 1]
我从旧梦走过长街

[Interlude]
啊 啊
oh oh

[Chorus]
月下长安 谁还未眠"""

    prepared = prepare_generation_lyrics(lyrics)

    assert "[Instrumental Intro]" in prepared
    assert "月色落在城墙边" not in prepared
    assert "风把灯影吹得很远" not in prepared
    assert "啊 啊" not in prepared
    assert "oh oh" not in prepared
    assert "我从旧梦走过长街" in prepared
    assert "月下长安 谁还未眠" in prepared


def test_detect_vocal_language_uses_english_when_no_cjk_text():
    assert detect_vocal_language("hello from the city lights", "English pop vocal") == "en"
    assert detect_vocal_language("月下长安", "Chinese guofeng pop") == "zh"
