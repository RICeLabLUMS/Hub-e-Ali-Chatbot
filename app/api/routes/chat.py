import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.core.config import settings
from app.core.dependencies import (
    get_embedder,
    get_openrouter_client,
    get_qdrant_client,
    get_reranker,
)
from app.services import memory
from app.services.ingestion.language_detector import detect_language
from app.services.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


class ChatRequest(BaseModel):
    message: str
    source: str | None = None   # accepted but currently unused (front-end filter UI)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@router.post("/chat")
async def chat(req: ChatRequest):
    question = req.message.strip()
    if not question:
        return {
            "answer": "Please ask a question.",
            "language_detected": "unknown",
            "source_label": "",
            "citations": [],
        }

    lang = detect_language(question)
    openrouter = get_openrouter_client()
    standalone = await openrouter.rewrite_query(question, memory.get_history())

    retriever = HybridRetriever(client=get_qdrant_client(), embedder=get_embedder())
    hits = await retriever.retrieve(
        standalone,
        top_k=settings.RETRIEVAL_TOP_K,
        language_filter=lang,
    )

    # If filtered retrieval came back empty, fall back to language-agnostic search
    if not hits and lang != "unknown":
        hits = await retriever.retrieve(
            standalone,
            top_k=settings.RETRIEVAL_TOP_K,
            language_filter=None,
        )

    reranked = get_reranker().rerank(standalone, hits, top_k=settings.RERANK_TOP_K)

    result = await openrouter.generate_answer(question, reranked, lang)
    answer = result.get("answer", "I don't know based on the provided sources.")
    citations = result.get("citations", [])

    memory.store_memory(question, answer)

    source_label = ""
    if reranked:
        top = reranked[0]
        source_label = f"{top.get('source', 'Unknown')} — page {top.get('page', 'N/A')}"

    return {
        "answer": answer,
        "language_detected": lang,
        "source_label": source_label,
        "citations": citations,
        # Legacy fields preserved for any existing UI binding
        "source_title": reranked[0].get("source") if reranked else "Unknown",
        "page": reranked[0].get("page") if reranked else "N/A",
    }
