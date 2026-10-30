"""Tests for the HeartMuLa service backend (heartmula_runner.py)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.heartmula_runner import _lyrics_text, _tags_from_prompt  # noqa: E402


def test_tags_from_prompt_keeps_relevant_tags():
    prompt = "chinese-style, pop, healing, female vocals, piano, guzheng, 80-100 BPM"
    tags = _tags_from_prompt(prompt)
    assert "pop" in tags
    assert "piano" in tags
    assert "guzheng" in tags
    assert "healing" in tags
    # BPM tokens are dropped (HeartMuLa tags do not understand them)
    assert "BPM" not in tags


def test_tags_from_prompt_fallback():
    tags = _tags_from_prompt("")
    assert "pop" in tags


def test_lyrics_text_instrumental():
    assert _lyrics_text("some lyrics", instrumental=True) == "[Instrumental]"


def test_lyrics_text_keeps_vocals():
    assert _lyrics_text("[Verse]\nHello", instrumental=False) == "[Verse]\nHello"
