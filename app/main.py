from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.utils import db
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    orphaned = db.reconcile_orphaned_jobs()
    if orphaned:
        logger.warning(
            "Marked %d job(s) FAILED on startup (orphaned by a previous process): %s",
            len(orphaned),
            ", ".join(orphaned),
        )
    yield


app = FastAPI(title="DD Automation", version="1.0.0", lifespan=lifespan)
app.include_router(router, prefix="/api")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
