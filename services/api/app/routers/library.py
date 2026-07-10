from fastapi import APIRouter, Query
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Job
from app.schemas import JobResponse

router = APIRouter(prefix="/library", tags=["library"])


def _job_to_response(job: Job) -> dict:
    resp = JobResponse(
        id=job.id,
        title=job.title,
        lyrics=job.lyrics,
        style_prompt=job.style_prompt,
        instrumental=job.instrumental,
        status=job.status,
        audio_path=job.audio_path,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
    data = resp.model_dump()
    if job.audio_path:
        data["audio_url"] = f"/audio/{job.id}.wav"
    return data


@router.get("")
def list_library(
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None, pattern="^(succeeded|failed|queued|running)$"),
):
    with SessionLocal() as session:
        stmt = select(Job).order_by(Job.created_at.desc())
        if status:
            stmt = stmt.where(Job.status == status)
        stmt = stmt.offset(offset).limit(limit)
        jobs = session.execute(stmt).scalars().all()
    return [_job_to_response(j) for j in jobs]
