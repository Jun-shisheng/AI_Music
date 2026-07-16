from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Job
from app.schemas import JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _job_to_response(job: Job) -> JobResponse:
    audio_url = f"/audio/{job.id}.wav" if job.audio_path else None
    return JobResponse(
        id=job.id,
        title=job.title,
        lyrics=job.lyrics,
        style_prompt=job.style_prompt,
        instrumental=job.instrumental,
        status=job.status,
        audio_path=job.audio_path,
        audio_url=audio_url,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_to_response(job)
