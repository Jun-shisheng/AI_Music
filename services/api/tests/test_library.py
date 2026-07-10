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
