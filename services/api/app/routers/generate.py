from fastapi import APIRouter

from app.db import SessionLocal
from app.models import Job
from app.schemas import GenerateRequest, GenerateResponse

router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("", response_model=GenerateResponse, status_code=201)
def create_generation(req: GenerateRequest):
    job = Job(
        title=req.title,
        lyrics=req.lyrics,
        style_prompt=req.style_prompt,
        instrumental=req.instrumental,
        status="queued",
    )
    with SessionLocal() as session:
        session.add(job)
        session.commit()
        session.refresh(job)
    return GenerateResponse(id=job.id, status=job.status)
