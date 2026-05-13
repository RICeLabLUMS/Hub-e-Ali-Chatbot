"""Singleton providers for heavy resources (models, clients, queue)."""

from functools import lru_cache
from qdrant_client import QdrantClient

from app.core.config import settings
from app.services.ingestion.embedder import BGEEmbedder
from app.services.ingestion.pdf_extractor import PDFExtractor
from app.services.retrieval.reranker import BGEReranker
from app.services.generation.openrouter_client import OpenRouterClient
from app.services.ingestion.job_queue import IngestionJobQueue


@lru_cache(maxsize=1)
def get_embedder() -> BGEEmbedder:
    return BGEEmbedder()


@lru_cache(maxsize=1)
def get_reranker() -> BGEReranker:
    return BGEReranker()


@lru_cache(maxsize=1)
def get_pdf_extractor() -> PDFExtractor:
    return PDFExtractor()


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
    )


@lru_cache(maxsize=1)
def get_openrouter_client() -> OpenRouterClient:
    return OpenRouterClient()


@lru_cache(maxsize=1)
def get_job_queue() -> IngestionJobQueue:
    return IngestionJobQueue()
