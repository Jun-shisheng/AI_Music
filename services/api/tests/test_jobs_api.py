from httpx import ASGITransport, AsyncClient
import pytest

from app.main import app


@pytest.mark.asyncio
async def test_create_generation_returns_queued():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/generate", json={
            "title": "测试歌曲",
            "lyrics": "[前奏]\n[主歌]\n歌词内容\n[副歌]\n副歌内容",
            "style_prompt": "pop, chinese, upbeat",
        })
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert len(data["id"]) == 12


@pytest.mark.asyncio
async def test_get_job_returns_details():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create_resp = await client.post("/generate", json={
            "title": "测试歌曲",
            "lyrics": "[Verse] test",
            "style_prompt": "rock",
        })
        job_id = create_resp.json()["id"]

        get_resp = await client.get(f"/jobs/{job_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["title"] == "测试歌曲"
    assert data["status"] == "queued"
    assert data["lyrics"] == "[Verse] test"


@pytest.mark.asyncio
async def test_get_nonexistent_job_returns_404():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/jobs/nonexist12")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_job_includes_audio_url_when_audio_exists():
    from app.db import SessionLocal
    from app.models import Job

    job = Job(
        title="Finished",
        lyrics="[Verse] done",
        style_prompt="pop",
        status="succeeded",
        audio_path="G:/AI_Music/outputs/manual.wav",
    )
    with SessionLocal() as session:
        session.add(job)
        session.commit()
        job_id = job.id

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/jobs/{job_id}")

    assert resp.status_code == 200
    assert resp.json()["audio_url"] == f"/audio/{job_id}.wav"
