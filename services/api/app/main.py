from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import init_db
from app.routers import arrange, jobs, generate, library

import os


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    from app.services.queue import start_worker, stop_worker
    await start_worker()
    yield
    await stop_worker()


app = FastAPI(title="AI Music API", version="0.1.0", lifespan=lifespan)

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
    path = os.path.join(settings.output_dir, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/health")
async def health():
    return {"status": "ok"}
