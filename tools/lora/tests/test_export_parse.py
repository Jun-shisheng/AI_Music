from export_dataset import parse_top_song_txt


SAMPLE = """Title: 霓虹雨巷
Score: 95
Category: sweet_female

[前奏]
霓虹亮起雨巷
[主歌1]
第一句歌词
第二句歌词
"""


def test_parse_top_song_txt():
    title, score, lyrics = parse_top_song_txt(SAMPLE)
    assert title == "霓虹雨巷"
    assert score == 95
    assert "[前奏]" in lyrics
    assert "第一句歌词" in lyrics
    assert "Title:" not in lyrics
