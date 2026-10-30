from fastapi import APIRouter

from app.db import SessionLocal
from app.models import Job
from app.schemas import GenerateRequest, GenerateResponse
from app.services.music_text import prepare_generation_lyrics
from app.services.queue import enqueue

router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("", response_model=GenerateResponse, status_code=201)
async def create_generation(req: GenerateRequest):
    duration_seconds = req.duration_seconds
    if duration_seconds is None and req.generation_type == "song":
        duration_seconds = 210
    lyrics = req.lyrics or ("[Instrumental]" if req.instrumental else "")
    if not req.instrumental:
        lyrics = prepare_generation_lyrics(lyrics)

    job = Job(
        title=req.title,
        lyrics=lyrics,
        style_prompt=req.style_prompt,
        instrumental=req.instrumental,
        generation_type=req.generation_type,
        duration_seconds=duration_seconds,
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
