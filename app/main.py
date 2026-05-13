import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import admin, chat, ingest
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

    yield

    logger.info("Shutting down...")
    await queue.stop()
    await get_openrouter_client().aclose()


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
