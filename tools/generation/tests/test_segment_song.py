"""Tests for segment_song.py lyrics splitting and crossfade joining."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segment_song import crossfade_join, split_lyrics  # noqa: E402


def test_split_lyrics_no_structure():
    lyrics = "line1\nline2\nline3\nline4\nline5\nline6\n"
    parts = split_lyrics(lyrics, 2)
    assert len(parts) == 2
    assert "line1" in parts[0]
    assert "line6" in parts[1]


def test_split_lyrics_by_sections():
    lyrics = """[主歌1]
aaa
bbb
[副歌1]
ccc
ddd
[主歌2]
eee
fff
[副歌2]
ggg
hhh
"""
    parts = split_lyrics(lyrics, 2)
    assert len(parts) >= 2
    joined = "\n".join(parts)
    for token in ("aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh"):
        assert token in joined, f"content lost: {token}"


def test_split_lyrics_content_preserved():
    lyrics = """[Verse 1]
aaa
bbb
[Chorus]
ccc
ddd
"""
    parts = split_lyrics(lyrics, 2)
    assert "aaa" in "".join(parts)
    assert "ddd" in "".join(parts)


def test_split_lyrics_keeps_order_no_duplicates():
    lyrics = """[A]
a1
a2
[B]
b1
b2
[C]
c1
c2
[D]
d1
d2
"""
    parts = split_lyrics(lyrics, 2)
    assert len(parts) == 2
    joined = "\n".join(parts)
    # every section header appears exactly once
    for h in ("[A]", "[B]", "[C]", "[D]"):
        assert joined.count(h) == 1, f"{h} count={joined.count(h)}"
    # order preserved
    assert joined.index("b1") > joined.index("a2")
    assert joined.index("d1") > joined.index("c2")


def test_crossfade_join_combines_wavs(tmp_path):
    import numpy as np
    import soundfile as sf

    sr = 48000
    segs = []
    for i, freq in enumerate((220, 330)):
        t = np.arange(sr * 3) / sr  # 3s segments, well above 2*fade
        wav = tmp_path / f"seg{i}.wav"
        sf.write(wav, np.stack([np.sin(2 * np.pi * freq * t)] * 2, axis=1), sr, subtype="PCM_16")
        segs.append(wav)
    out = tmp_path / "joined.wav"
    crossfade_join(segs, out, fade_s=0.5)
    y, _ = sf.read(out, always_2d=True)
    mono = y.mean(axis=1)
    # 3+3-0.5 overlap = 5.5s
    assert 5.2 < len(mono) / sr < 5.8, f"joined wrong length: {len(mono)/sr:.2f}s"
    # no silence gap at the seam (t around 2.5-3.5s)
    seam = mono[int(2.5 * sr):int(3.5 * sr)]
    assert float(np.sqrt(np.mean(seam**2))) > 0.05, "silence gap at seam"
