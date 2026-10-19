"""Instrumental jobs must never forward stale vocal lyrics to the backend."""
import json

import httpx
import pytest

from app.config import settings
from app.main import app
from app.services import acestep_runner as runner


@pytest.mark.asyncio
@pytest.mark.parametrize("generation_type,instrumental", [
    ("song", True), ("bgm_loop", False), ("sfx", False),
])
async def test_instrumental_job_stores_effective_lyrics(generation_type, instrumental):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post("/generate", json={
            "lyrics": "[Verse]\nThese words must not be sung",
            "style_prompt": "piano",
            "generation_type": generation_type,
            "instrumental": instrumental,
        })
        assert created.status_code == 201
        job = (await client.get(f"/jobs/{created.json()['id']}")).json()
        library = (await client.get("/library")).json()

    assert job["instrumental"] is True
    assert job["lyrics"] == "[Instrumental]"
    assert next(item for item in library if item["id"] == job["id"])["lyrics"] == "[Instrumental]"


@pytest.mark.asyncio
@pytest.mark.parametrize("generation_type,instrumental,expected_lyrics", [
    ("song", False, "[Verse]\nKeep these lyrics"),
    ("song", True, "[Instrumental]"),
    ("bgm_loop", False, "[Instrumental]"),
    ("sfx", False, "[Instrumental]"),
])
async def test_effective_lyrics_reach_acestep_api(
    monkeypatch, tmp_path, generation_type, instrumental, expected_lyrics
):
    submitted = {}
    # A complete tiny PCM WAV, served by a local HTTP mock rather than a model.
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 80)
    audio = buffer.getvalue()

    def handle(request):
        if request.url.path == "/release_task":
            submitted.update(json.loads(request.content))
            return httpx.Response(200, json={"data": {"task_id": "test-task"}})
        if request.url.path == "/query_result":
            return httpx.Response(200, json={"data": [{
                "task_id": "test-task", "status": 1,
                "result": json.dumps([{"file": "/v1/audio?path=test.wav"}]),
            }]})
        assert request.url.path == "/v1/audio"
        return httpx.Response(200, content=audio, headers={"content-type": "audio/wav"})

    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=transport, **kwargs
    ))

    async def no_wait(_seconds):
        pass

    monkeypatch.setattr(runner.asyncio, "sleep", no_wait)
    monkeypatch.setattr(runner, "BACKEND", "api")
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    target = tmp_path / "output.wav"
    result = await runner.generate(
        lyrics="[Verse]\nKeep these lyrics", style_prompt="piano",
        output_path=str(target), generation_type=generation_type,
        instrumental=instrumental,
    )

    assert submitted["lyrics"] == expected_lyrics
    assert submitted["audio_format"] == "wav"
    assert result == target
    assert target.read_bytes() == audio
