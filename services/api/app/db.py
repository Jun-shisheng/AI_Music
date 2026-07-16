from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

DATABASE_URL = settings.database_url

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def init_db():
    Base.metadata.create_all(bind=engine)
    _ensure_job_columns()


def _ensure_job_columns():
    if not DATABASE_URL.startswith("sqlite"):
        return

    columns = {
        "generation_type": "VARCHAR(40) DEFAULT 'song' NOT NULL",
        "duration_seconds": "INTEGER",
        "loopable": "BOOLEAN DEFAULT 0 NOT NULL",
        "variation_label": "VARCHAR(80)",
    }
    with engine.begin() as conn:
        table_exists = conn.execute(
            text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'")
        ).scalar()
        if not table_exists:
            return
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(jobs)"))}
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE jobs ADD COLUMN {name} {ddl}"))


def drop_db():
    Base.metadata.drop_all(bind=engine)
