from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import init_db
from app.routers import jobs, generate


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Music API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)
app.include_router(generate.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


