from httpx import ASGITransport, AsyncClient
import pytest

from app.main import app


@pytest.mark.asyncio
async def test_library_returns_empty_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/library")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_library_returns_created_jobs():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a job first
        await client.post("/generate", json={
            "title": "Library Test",
            "lyrics": "[Verse] test",
            "style_prompt": "pop",
        })

        resp = await client.get("/library")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    titles = [j["title"] for j in data]
    assert "Library Test" in titles


@pytest.mark.asyncio
async def test_library_filter_by_status():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/library?status=queued")
    assert resp.status_code == 200
    for job in resp.json():
        assert job["status"] == "queued"


@pytest.mark.asyncio
async def test_library_includes_audio_url_for_completed_jobs():
    from app.db import SessionLocal
    from app.models import Job

    job = Job(
        title="Playable",
        lyrics="[Verse] listen",
        style_prompt="piano",
        status="succeeded",
        audio_path="outputs/playable.wav",
    )
    with SessionLocal() as session:
        session.add(job)
        session.commit()
        job_id = job.id

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/library?status=succeeded")

    assert resp.status_code == 200
    item = next(j for j in resp.json() if j["id"] == job_id)
    assert item["audio_url"] == f"/audio/{job_id}.wav"


@pytest.mark.asyncio
async def test_library_returns_game_asset_metadata():
    from app.db import SessionLocal
    from app.models import Job

    job = Job(
        title="Impact Hit",
        lyrics="[Instrumental]",
        style_prompt="game impact sound, punchy transient",
        instrumental=True,
        generation_type="sfx",
        duration_seconds=10,
        loopable=False,
        variation_label="C",
        status="queued",
    )
    with SessionLocal() as session:
        session.add(job)
        session.commit()
        job_id = job.id

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/library")

    assert resp.status_code == 200
    item = next(j for j in resp.json() if j["id"] == job_id)
    assert item["generation_type"] == "sfx"
    assert item["duration_seconds"] == 10
    assert item["variation_label"] == "C"
