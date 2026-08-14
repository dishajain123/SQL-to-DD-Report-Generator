from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.utils import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="DD Automation", version="1.0.0", lifespan=lifespan)
app.include_router(router, prefix="/api")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
