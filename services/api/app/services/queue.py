"""Simple in-process job queue for MVP single-machine deployment."""

import asyncio
import logging
from pathlib import Path

from app.config import settings
from app.db import SessionLocal
from app.models import Job
from app.services.acestep_runner import generate as run_acestep

logger = logging.getLogger(__name__)

_queue: asyncio.Queue[str] = asyncio.Queue()
_worker_task: asyncio.Task | None = None


async def enqueue(job_id: str):
    await _queue.put(job_id)


async def start_worker():
    global _worker_task
    if _worker_task is not None:
        return
    _worker_task = asyncio.create_task(_run_worker())
    logger.info("Worker started")


async def stop_worker():
    global _worker_task
    if _worker_task:
        _worker_task.cancel()
        _worker_task = None
    logger.info("Worker stopped")


async def _run_worker():
    while True:
        job_id = await _queue.get()
        try:
            await _process_job(job_id)
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
        finally:
            _queue.task_done()


async def _process_job(job_id: str):
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        job.status = "running"
        session.commit()

        try:
            output_name = f"{job_id}.wav"
            output_path = str(Path(settings.output_dir) / output_name)

            await run_acestep(
                lyrics=job.lyrics,
                style_prompt=job.style_prompt,
                output_path=output_path,
                instrumental=job.instrumental,
            )

            job.status = "succeeded"
            job.audio_path = output_path
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.exception(f"Job {job_id} generation failed")

        session.commit()
