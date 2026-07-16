from fastapi import APIRouter

from app.db import SessionLocal
from app.models import Job
from app.schemas import GenerateRequest, GenerateResponse
from app.services.queue import enqueue

router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("", response_model=GenerateResponse, status_code=201)
async def create_generation(req: GenerateRequest):
    job = Job(
        title=req.title,
        lyrics=req.lyrics or ("[Instrumental]" if req.instrumental else ""),
        style_prompt=req.style_prompt,
        instrumental=req.instrumental,
        generation_type=req.generation_type,
        duration_seconds=req.duration_seconds,
        loopable=req.loopable,
        variation_label=req.variation_label,
        status="queued",
    )
    with SessionLocal() as session:
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    await enqueue(job_id)
    return GenerateResponse(id=job_id, status="queued")
