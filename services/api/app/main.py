from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import settings
from app.db import init_db
from app.routers import arrange, jobs, generate, library

from pathlib import Path


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    from app.services.queue import start_worker, stop_worker
    await start_worker()
    try:
        yield
    finally:
        await stop_worker()


app = FastAPI(title="AI Music API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(arrange.router)
app.include_router(jobs.router)
app.include_router(generate.router)
app.include_router(library.router)


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve one generated WAV, and nothing outside OUTPUT_DIR.

    `filename` is user-controlled. Joining it onto OUTPUT_DIR is not enough:
    on Windows an absolute segment (`C:\\Windows\\win.ini`) discards the base
    directory entirely, and `..\\` segments escape it. Only a bare file name
    that resolves to a direct child of the output directory is served.
    """
    if not filename or Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="Audio file not found")

    output_dir = Path(settings.output_dir).resolve()
    path = (output_dir / filename).resolve()
    if path.parent != output_dir or path.suffix.lower() != ".wav" or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/health")
async def health():
    return {"status": "ok"}
