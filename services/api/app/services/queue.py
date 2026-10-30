"""Simple in-process job queue for MVP single-machine deployment."""

import asyncio
import logging
import subprocess
from pathlib import Path

from app.config import settings
from app.db import SessionLocal
from app.models import Job
from app.services.acestep_runner import generate as run_acestep
from app.services.heartmula_runner import generate as run_heartmula

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TOOLS_VENV_PYTHON = PROJECT_ROOT / "tools" / "vocal_repair" / ".venv" / "Scripts" / "python.exe"
HUMANIZE_SCRIPT = PROJECT_ROOT / "tools" / "vocal_repair" / "humanize_vocal.py"


async def humanize_generated(output_path: Path) -> Path:
    """Post-process a generated song: demucs -> frame repair -> de_buzz EQ -> remix.

    Falls back to the raw output on any failure so generation never breaks.
    """
    if not TOOLS_VENV_PYTHON.exists():
        logger.warning("vocal_repair venv missing at %s; skipping vocal humanization", TOOLS_VENV_PYTHON)
        return output_path
    work_dir = PROJECT_ROOT / "outputs" / "humanize_work" / output_path.stem
    cmd = [
        str(TOOLS_VENV_PYTHON),
        str(HUMANIZE_SCRIPT),
        str(output_path),
        "--variant", settings.vocal_humanize_variant,
        "--output", str(output_path),
        "--work-dir", str(work_dir),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=1800)
        if proc.returncode != 0:
            logger.error("vocal humanization failed: %s", stderr.decode()[:2000])
            return output_path
        logger.info("vocal humanization applied: %s", stdout.decode().strip()[-200:])
    except Exception as exc:  # noqa: BLE001 - humanization must never break generation
        logger.warning("vocal humanization error (falling back to raw output): %s", exc)
        return output_path
    return output_path

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

            if settings.generation_backend == "heartmula":
                # Redesigned source-clean path: HeartMuLa -> native 48k WAV.
                # NO post processing (repair chain abandoned).
                await run_heartmula(
                    lyrics=job.lyrics,
                    style_prompt=job.style_prompt,
                    output_path=output_path,
                    instrumental=job.instrumental,
                    duration_seconds=job.duration_seconds,
                    loopable=job.loopable,
                )
            else:
                await run_acestep(
                    lyrics=job.lyrics,
                    style_prompt=job.style_prompt,
                    output_path=output_path,
                    instrumental=job.instrumental,
                    duration_seconds=job.duration_seconds,
                    loopable=job.loopable,
                )
                if settings.vocal_humanize_enabled and not job.instrumental:
                    output_path = await humanize_generated(Path(output_path))

            job.status = "succeeded"
            job.audio_path = str(output_path)
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.exception(f"Job {job_id} generation failed")

        session.commit()
