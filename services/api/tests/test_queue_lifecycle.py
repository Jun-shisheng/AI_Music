"""Queue restart, duplicate delivery, and interrupted generation regressions."""

import asyncio
from pathlib import Path

import pytest

from app.config import settings
from app.db import SessionLocal
from app.models import Job
from app.services import queue


@pytest.fixture
async def isolated_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(queue, "_queue", asyncio.Queue())
    monkeypatch.setattr(queue, "_worker_task", None)
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    yield
    await queue.stop_worker()


def _create_job(status="queued"):
    with SessionLocal() as session:
        job = Job(title="Queue lifecycle", lyrics="test", style_prompt="pop",
                  status=status)
        session.add(job)
        session.commit()
        return job.id


def _job_state(job_id):
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        return job.status, job.error, job.audio_path


@pytest.mark.parametrize("status", ["succeeded", "failed", "running"])
async def test_duplicate_delivery_does_not_restart_nonqueued_job(
    status, isolated_worker, monkeypatch,
):
    job_id = _create_job(status)

    async def must_not_run(**kwargs):
        pytest.fail("A non-queued job was sent to the generator")

    monkeypatch.setattr(queue, "run_acestep", must_not_run)
    await queue._process_job(job_id)
    assert _job_state(job_id)[0] == status


async def test_overlapping_delivery_claims_job_once(isolated_worker, monkeypatch):
    job_id = _create_job()
    entered = asyncio.Event()
    finish = asyncio.Event()
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        entered.set()
        await finish.wait()
        Path(kwargs["output_path"]).write_bytes(b"audio")

    monkeypatch.setattr(queue, "run_acestep", generate)
    first = asyncio.create_task(queue._process_job(job_id))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        await queue._process_job(job_id)
        assert _job_state(job_id)[0] == "running"
        finish.set()
        await asyncio.wait_for(first, timeout=2)
        await queue._process_job(job_id)
    finally:
        if not first.done():
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

    assert len(calls) == 1
    assert _job_state(job_id) == (
        "succeeded", None, calls[0]["output_path"],
    )


async def test_stop_fails_running_and_restart_preserves_queued(
    isolated_worker, monkeypatch,
):
    running_id = _create_job()
    queued_id = _create_job()
    entered = asyncio.Event()
    calls = []

    async def generate(**kwargs):
        output = Path(kwargs["output_path"])
        calls.append(output.stem)
        if output.stem == running_id:
            entered.set()
            await asyncio.Event().wait()
        output.write_bytes(b"audio")

    monkeypatch.setattr(queue, "run_acestep", generate)
    # Enqueue and recovery both used to leave the same IDs in the old queue.
    await queue.enqueue(running_id)
    await queue.enqueue(queued_id)
    await queue.start_worker()
    await asyncio.wait_for(entered.wait(), timeout=2)
    old_queue = queue._queue
    await queue.stop_worker()

    assert _job_state(running_id) == ("failed", queue.RESTART_ERROR, None)
    assert _job_state(queued_id) == ("queued", None, None)

    await queue.start_worker()
    assert queue._queue is not old_queue
    await asyncio.wait_for(queue._queue.join(), timeout=2)
    assert calls == [running_id, queued_id]
    assert _job_state(queued_id)[0] == "succeeded"


def test_worker_restarts_on_a_new_event_loop(monkeypatch):
    monkeypatch.setattr(queue, "_queue", asyncio.Queue())
    monkeypatch.setattr(queue, "_worker_task", None)

    async def run_lifespan():
        await queue.start_worker()
        worker = queue._worker_task
        try:
            # Let the empty queue bind to this loop inside queue.get().
            await asyncio.sleep(0)
            assert not worker.done()
        finally:
            await queue.stop_worker()

    asyncio.run(run_lifespan())
    asyncio.run(run_lifespan())


@pytest.mark.parametrize("write_empty_file", [False, True])
async def test_missing_or_empty_output_fails_job(
    write_empty_file, isolated_worker, monkeypatch,
):
    job_id = _create_job()

    async def generate(**kwargs):
        if write_empty_file:
            Path(kwargs["output_path"]).touch()

    monkeypatch.setattr(queue, "run_acestep", generate)
    await queue._process_job(job_id)

    status, error, audio_path = _job_state(job_id)
    assert status == "failed"
    assert "non-empty audio file" in error
    assert audio_path is None
