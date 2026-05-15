import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import admin, chat, ingest
from app.core.config import settings
from app.core.dependencies import (
    get_embedder,
    get_job_queue,
    get_openrouter_client,
    get_qdrant_client,
    get_reranker,
)
from app.services.qdrant_setup import setup_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming models (bge-m3, bge-reranker-v2-m3)...")
    get_embedder()
    get_reranker()

    logger.info("Ensuring Qdrant collection exists...")
    setup_collection(get_qdrant_client(), recreate=False)

    logger.info("Starting ingestion job queue...")
    queue = get_job_queue()
    await queue.start()

    wp_scheduler = _maybe_start_wordpress_scheduler()

    yield

    logger.info("Shutting down...")
    if wp_scheduler is not None:
        wp_scheduler.shutdown(wait=False)
    await queue.stop()
    await get_openrouter_client().aclose()


def _maybe_start_wordpress_scheduler():
    """Start APScheduler-driven WordPress sync if enabled. Returns the scheduler or None."""
    if not (settings.WORDPRESS_SYNC_ENABLED and settings.WORDPRESS_URL):
        return None

    from datetime import datetime, timedelta
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.interval import IntervalTrigger

    from app.services.ingestion.wordpress_sync import WordPressSync

    wp_sync = WordPressSync()
    scheduler = AsyncIOScheduler()
    # wp_sync.sync is synchronous (httpx sync client) - AsyncIOScheduler runs
    # sync callables on its default ThreadPoolExecutor, so the event loop is
    # never blocked.
    # Kick off shortly after startup so model warm-up has time to settle,
    # then run on the configured interval. max_instances=1 + coalesce ensures
    # overlapping runs are skipped rather than queued.
    scheduler.add_job(
        wp_sync.sync,
        trigger=IntervalTrigger(minutes=settings.WORDPRESS_SYNC_INTERVAL_MINUTES),
        id="wordpress_sync",
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now() + timedelta(seconds=30),
    )
    scheduler.start()
    logger.info(
        f"WordPress sync scheduler started (interval={settings.WORDPRESS_SYNC_INTERVAL_MINUTES}m)"
    )
    return scheduler


app = FastAPI(title="HubeAli Multilingual RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Root chat router (preserves GET / and POST /chat for the existing front-end)
app.include_router(chat.router)
# API routers (ingestion & admin live under /api)
app.include_router(ingest.router, prefix="/api")
app.include_router(admin.router, prefix="/api")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
