from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    title: str = "Untitled"
    lyrics: str = ""
    style_prompt: str = ""
    instrumental: bool = False
    generation_type: Literal["song", "bgm_loop", "sfx"] = "song"
    duration_seconds: int | None = Field(default=None, ge=10, le=600)
    loopable: bool = False
    variation_label: str | None = Field(default=None, max_length=80)


class GenerateResponse(BaseModel):
    id: str
    status: str


class JobResponse(BaseModel):
    id: str
    title: str
    lyrics: str
    style_prompt: str
    instrumental: bool
    generation_type: str = "song"
    duration_seconds: int | None = None
    loopable: bool = False
    variation_label: str | None = None
    status: str
    audio_path: str | None = None
    audio_url: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
