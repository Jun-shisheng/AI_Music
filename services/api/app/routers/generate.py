from fastapi import APIRouter

from app.db import SessionLocal
from app.models import Job
from app.schemas import GenerateRequest, GenerateResponse
from app.services.acestep_runner import resolve_request
from app.services.queue import enqueue

router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("", response_model=GenerateResponse, status_code=201)
async def create_generation(req: GenerateRequest):
    # Store the resolved request, not the raw flags: a bgm_loop is always
    # instrumental and loopable, and an sfx has a 10 s default. Keeping the raw
    # values made /jobs and /library disagree with docs/api.md.
    resolved = resolve_request(
        req.generation_type,
        req.instrumental,
        req.duration_seconds,
        req.loopable,
    )
    job = Job(
        title=req.title,
        lyrics="[Instrumental]" if resolved["instrumental"] else req.lyrics,
        style_prompt=req.style_prompt,
        instrumental=resolved["instrumental"],
        generation_type=req.generation_type,
        duration_seconds=resolved["duration_seconds"],
        loopable=resolved["loopable"],
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
