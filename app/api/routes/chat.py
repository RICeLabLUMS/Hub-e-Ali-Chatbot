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
    model_citations = result.get("citations", [])

    memory.store_memory(question, answer)

    # Enrich the model's chunk_id citations with display metadata from the
    # reranked payloads. Anything the LLM cited but wasn't in context is dropped.
    chunk_index = {c.get("chunk_id"): c for c in reranked if c.get("chunk_id")}
    citations = []
    for cit in model_citations:
        chunk_id = cit.get("chunk_id")
        ref = chunk_index.get(chunk_id)
        if not ref:
            continue
        citations.append(_build_citation(ref))

    # Fall back to the top reranked hit if the LLM produced no usable citations.
    if not citations and reranked:
        citations.append(_build_citation(reranked[0]))

    top_label = citations[0]["label"] if citations else ""
    top_url = citations[0]["url"] if citations else None

    return {
        "answer": answer,
        "language_detected": lang,
        "source_label": top_label,
        "source_url": top_url,
        "citations": citations,
        # Legacy fields preserved for any existing UI binding
        "source_title": citations[0]["title"] if citations else "Unknown",
        "page": citations[0]["page"] if citations else "N/A",
    }


def _build_citation(chunk: dict) -> dict:
    """Shape one chunk's payload into a display-ready citation dict."""
    title = chunk.get("title") or chunk.get("source") or "Unknown"
    content_type = chunk.get("content_type") or ""
    page = chunk.get("page")
    url = chunk.get("url")

    parts = [title]
    if content_type:
        suffix = content_type
        if content_type.upper() == "PDF" and page:
            suffix = f"{content_type}, p. {page}"
        parts.append(suffix)
    elif page is not None:
        parts.append(f"p. {page}")
    label = " — ".join(parts)

    return {
        "chunk_id": chunk.get("chunk_id"),
        "title": title,
        "content_type": content_type,
        "page": page,
        "url": url,
        "source": chunk.get("source"),
        "label": label,
    }
