"""Regression tests for defects found in the 2026-10-19 audit."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_audio_serving_is_confined_to_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    (tmp_path / "ok.wav").write_bytes(b"RIFF....WAVE")

    secret = tmp_path.parent / "secret.txt"
    secret.write_text("do not serve", encoding="utf-8")

    async with _client() as client:
        ok = await client.get("/audio/ok.wav")
        assert ok.status_code == 200
        assert ok.headers["content-type"] == "audio/wav"

        # Absolute Windows path used to override OUTPUT_DIR via os.path.join.
        escaped = await client.get("/audio/C:%5CWindows%5Cwin.ini")
        assert escaped.status_code == 404

        for name in ("..%5Csecret.txt", "..%2Fsecret.txt", "sub%5C..%5Csecret.txt"):
            resp = await client.get(f"/audio/{name}")
            assert resp.status_code == 404

        missing = await client.get("/audio/nope.wav")
        assert missing.status_code == 404


@pytest.mark.asyncio
async def test_bgm_loop_is_stored_with_resolved_flags():
    async with _client() as client:
        created = await client.post("/generate", json={
            "title": "loop",
            "lyrics": "[Instrumental]",
            "style_prompt": "calm",
            "generation_type": "bgm_loop",
        })
        assert created.status_code == 201
        job = (await client.get(f"/jobs/{created.json()['id']}")).json()

    assert job["instrumental"] is True
    assert job["loopable"] is True


@pytest.mark.asyncio
async def test_sfx_is_stored_with_its_default_duration():
    async with _client() as client:
        created = await client.post("/generate", json={
            "title": "hit",
            "lyrics": "",
            "style_prompt": "impact",
            "generation_type": "sfx",
        })
        job = (await client.get(f"/jobs/{created.json()['id']}")).json()

    assert job["instrumental"] is True
    assert job["duration_seconds"] == 10
    assert job["lyrics"] == "[Instrumental]"


@pytest.mark.asyncio
async def test_library_rejects_non_positive_limit():
    async with _client() as client:
        assert (await client.get("/library?limit=0")).status_code == 422
        assert (await client.get("/library?limit=-1")).status_code == 422


@pytest.mark.asyncio
async def test_queue_recovery_fails_running_and_requeues_queued():
    from app.db import SessionLocal
    from app.models import Job
    from app.services.queue import RESTART_ERROR, _recover_jobs

    with SessionLocal() as session:
        running = Job(title="interrupted", lyrics="x", style_prompt="pop",
                      status="running")
        queued = Job(title="waiting", lyrics="x", style_prompt="pop",
                     status="queued")
        session.add_all([running, queued])
        session.commit()
        running_id, queued_id = running.id, queued.id

    assert _recover_jobs() == [queued_id]

    with SessionLocal() as session:
        assert session.get(Job, running_id).status == "failed"
        assert session.get(Job, running_id).error == RESTART_ERROR
        assert session.get(Job, queued_id).status == "queued"
