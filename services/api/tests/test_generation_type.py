"""Tests for the generation-type contract: song / bgm_loop / sfx.

The front end has always offered the three types, and the job row has always
stored the choice -- but the value never reached the generator, so asking for a
sound effect produced exactly what a song produced. These tests pin the mapping
between the API contract and the request that is actually sent to the model.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.services import acestep_runner as runner
from app.services.acestep_runner import SFX_DEFAULT_SECONDS, resolve_request


def test_song_keeps_the_caller_s_choices():
    req = resolve_request("song", instrumental=False, duration_seconds=180,
                          loopable=False)
    assert req["instrumental"] is False
    assert req["duration_seconds"] == 180
    assert req["loopable"] is False
    assert req["prompt_bits"] == []


def test_song_asked_for_instrumental_says_no_vocals():
    req = resolve_request("song", instrumental=True, duration_seconds=None,
                          loopable=False)
    assert "no vocals" in req["prompt_bits"]


def test_bgm_loop_is_instrumental_and_seamless():
    req = resolve_request("bgm_loop", instrumental=False, duration_seconds=None,
                          loopable=False)
    assert req["instrumental"] is True
    assert req["loopable"] is True
    assert "game background music" in req["prompt_bits"]
    assert "seamless loop" in req["prompt_bits"]


def test_sfx_is_short_and_explicitly_not_music():
    req = resolve_request("sfx", instrumental=False, duration_seconds=None,
                          loopable=False)
    assert req["instrumental"] is True
    assert req["duration_seconds"] == SFX_DEFAULT_SECONDS
    assert "sound effect" in req["prompt_bits"]
    assert "no music" in req["prompt_bits"]


def test_sfx_keeps_a_duration_the_caller_chose():
    assert resolve_request("sfx", False, 45, False)["duration_seconds"] == 45


def test_an_unknown_type_is_rejected():
    with pytest.raises(ValueError, match="generation_type"):
        resolve_request("podcast", instrumental=False, duration_seconds=None,
                        loopable=False)


def test_generate_hands_the_resolved_request_to_the_backend(monkeypatch, tmp_path):
    """The plumbing between the queue and the generator, which is where a type
    can quietly go missing: stub the backend and assert what it was asked for."""
    seen: dict = {}

    async def fake_backend(lyrics, style_prompt, output_path, instrumental,
                           duration_seconds, prompt_bits):
        seen.update(instrumental=instrumental,
                    duration_seconds=duration_seconds, prompt_bits=prompt_bits)
        return Path(output_path)

    monkeypatch.setattr(runner, "_ensure_output_dir", lambda: None)
    monkeypatch.setattr(runner, "BACKEND", "subprocess")
    monkeypatch.setattr(runner, "_generate_subprocess", fake_backend)

    out = asyncio.run(runner.generate(
        lyrics="", style_prompt="epic whoosh", output_path=str(tmp_path / "o.wav"),
        generation_type="sfx"))

    assert out == tmp_path / "o.wav"
    assert seen["instrumental"] is True
    assert seen["duration_seconds"] == SFX_DEFAULT_SECONDS
    assert "sound effect" in seen["prompt_bits"]
