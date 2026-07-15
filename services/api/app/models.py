import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=_new_id)
    title: Mapped[str] = mapped_column(String(200), default="Untitled")
    lyrics: Mapped[str] = mapped_column(Text, default="")
    style_prompt: Mapped[str] = mapped_column(Text, default="")
    instrumental: Mapped[bool] = mapped_column(default=False)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
