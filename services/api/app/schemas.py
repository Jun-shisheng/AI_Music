from datetime import datetime

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    title: str = "Untitled"
    lyrics: str = ""
    style_prompt: str = ""
    instrumental: bool = False


class GenerateResponse(BaseModel):
    id: str
    status: str


class JobResponse(BaseModel):
    id: str
    title: str
    lyrics: str
    style_prompt: str
    instrumental: bool
    status: str
    audio_path: str | None = None
    audio_url: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
