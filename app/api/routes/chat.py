import logging
import time
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Request
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


# ----- simple in-process per-IP rate limit -----
# Keeps a sliding window of recent request timestamps per client IP. Cheap and
# stateless beyond this dict; for production with multiple workers you'd want
# Redis-backed slowapi or similar.
_RATE_LIMIT_LOG: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=settings.CHAT_RATE_LIMIT_PER_MINUTE))


def _rate_limited(client_ip: str) -> bool:
    cap = settings.CHAT_RATE_LIMIT_PER_MINUTE
    if cap <= 0:
        return False
    now = time.time()
    window = _RATE_LIMIT_LOG[client_ip]
    # Discard timestamps older than 60s.
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= cap:
        return True
    window.append(now)
    return False


# UI labels -> Qdrant content_type values. Anything not in this map (including
# 'all', None, '') means "no filter".
SOURCE_TO_CONTENT_TYPES: dict[str, list[str]] = {
    "books": ["PDF"],
    "pdfs": ["PDF"],
    "articles": ["Article"],
    "pages": ["Page"],
}


class ChatRequest(BaseModel):
    message: str
    source: str | None = None       # UI filter: 'all' | 'books' | 'articles' | 'pages'
    session_id: str | None = None   # opaque per-user session id


class ResetRequest(BaseModel):
    session_id: str


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # backend_url is injected into the page as window.HA_CONFIG.backendUrl so
    # the front-end can call this API cross-origin if it's embedded elsewhere.
    # Empty string -> JS uses same-origin relative URLs (default deployment).
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "backend_url": (settings.PUBLIC_BACKEND_URL or "").rstrip("/"),
        },
    )


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    client_ip = request.client.host if request.client else "anon"
    if _rate_limited(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({settings.CHAT_RATE_LIMIT_PER_MINUTE}/min). Slow down a bit.",
        )

    question = req.message.strip()
    # New clients get a fresh session id back. Existing clients echo theirs.
    session_id = req.session_id or memory.new_session_id()

    if not question:
        return {
            "answer": "Please ask a question.",
            "language_detected": "unknown",
            "source_label": "",
            "citations": [],
            "session_id": session_id,
        }

    lang = detect_language(question)
    openrouter = get_openrouter_client()
    standalone = await openrouter.rewrite_query(question, memory.get_history(session_id))

    retriever = HybridRetriever(client=get_qdrant_client(), embedder=get_embedder())

    # Resolve the UI source filter to a content_type list (or None for 'all').
    content_type_filter = SOURCE_TO_CONTENT_TYPES.get((req.source or "").lower())

    hits = await retriever.retrieve(
        standalone,
        top_k=settings.RETRIEVAL_TOP_K,
        language_filter=lang,
        content_type_filter=content_type_filter,
    )

    # If the language-filtered pass returned too few candidates for the
    # reranker to do meaningful work, augment with a language-agnostic pass
    # and dedupe by chunk_id. This fixes mixed-language corpora where the
    # same document's chunks end up tagged with different languages (Arabic
    # verses + English commentary in the same article, etc).
    # We keep the content_type filter on both passes - that's user intent.
    min_useful = max(settings.RERANK_TOP_K * 2, 8)
    if len(hits) < min_useful and lang != "unknown":
        extra = await retriever.retrieve(
            standalone,
            top_k=settings.RETRIEVAL_TOP_K,
            language_filter=None,
            content_type_filter=content_type_filter,
        )
        seen_ids = {h["chunk_id"] for h in hits if h.get("chunk_id")}
        hits.extend(h for h in extra if h.get("chunk_id") and h["chunk_id"] not in seen_ids)

    reranked = get_reranker().rerank(standalone, hits, top_k=settings.RERANK_TOP_K)

    # One-line view of what's about to be sent to the LLM. Lets you correlate
    # "I don't know" answers with shallow/wrong reranked chunks.
    if reranked:
        top_summary = ", ".join(
            f"{(c.get('title') or c.get('source') or '?')[:30]}"
            f"[{c.get('content_type') or '?'}{':p' + str(c['page']) if c.get('page') else ''}]"
            f"@{c.get('rerank_score', 0):.2f}"
            for c in reranked[:5]
        )
    else:
        top_summary = "<none>"
    logger.info(
        f"[chat] lang={lang} standalone={standalone[:80]!r} "
        f"hits={len(hits)} reranked={len(reranked)} top: {top_summary}"
    )

    result = await openrouter.generate_answer(question, reranked, lang)
    answer = result.get("answer", "I don't know based on the provided sources.")
    model_citations = result.get("citations", [])

    # store_memory filters 'I don't know' answers internally - no need to
    # gate here.
    memory.store_memory(session_id, question, answer)

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
        "session_id": session_id,
        # Legacy fields preserved for any existing UI binding
        "source_title": citations[0]["title"] if citations else "Unknown",
        "page": citations[0]["page"] if citations else "N/A",
    }


@router.post("/chat/reset")
async def chat_reset(req: ResetRequest):
    """Clear a session's conversation history."""
    existed = memory.reset(req.session_id)
    return {"reset": existed, "session_id": req.session_id}


@router.get("/chat/stats")
async def chat_stats():
    """Diagnostics on in-memory session storage."""
    return memory.stats()


def _build_citation(chunk: dict) -> dict:
    """
    Shape one chunk's payload into a display-ready citation dict.

    Label format prefers numeric anchors when available:
      "AL-ANFAAL Chapter 8 — Article · §VERSE 1 · Quran 8:1"
      "Al-Kafi — PDF · Vol. 8, p. 42"
      "Nahjul Balagha — PDF · §Sermon 17 · p. 88"
      "Tawheed concept — Article"             (no numbers found)
    """
    title = chunk.get("title") or chunk.get("source") or "Unknown"
    content_type = chunk.get("content_type") or ""
    page = chunk.get("page")
    url = chunk.get("url")

    volume = chunk.get("volume")
    section = chunk.get("section_title")
    refs = chunk.get("refs_quran") or []
    hadith = chunk.get("hadith_refs") or []
    chapter_num = chunk.get("chapter_num")
    verse_range = chunk.get("verse_range")

    parts: list[str] = [title]

    # Type + page/volume cluster
    if content_type:
        type_bits = [content_type]
        if volume is not None:
            type_bits.append(f"Vol. {volume}")
        if page is not None and content_type.upper() == "PDF":
            type_bits.append(f"p. {page}")
        parts.append(", ".join(type_bits))
    elif page is not None:
        parts.append(f"p. {page}")

    # Section / chapter / verse range
    if section:
        parts.append(f"§{section}")
    elif chapter_num is not None and verse_range:
        parts.append(f"Ch. {chapter_num} v. {verse_range}")
    elif chapter_num is not None:
        parts.append(f"Ch. {chapter_num}")

    # Quran references (cap at 3 to keep labels readable)
    if refs:
        refs_display = refs[:3]
        more = "" if len(refs) <= 3 else f" +{len(refs) - 3}"
        parts.append(f"Quran {', '.join(refs_display)}{more}")

    # Hadith / sermon / letter refs (also capped at 3).
    if hadith:
        hadith_display = hadith[:3]
        h_more = "" if len(hadith) <= 3 else f" +{len(hadith) - 3}"
        parts.append(f"{', '.join(hadith_display)}{h_more}")

    label = " — ".join(parts)

    return {
        "chunk_id": chunk.get("chunk_id"),
        "title": title,
        "content_type": content_type,
        "page": page,
        "url": url,
        "source": chunk.get("source"),
        "volume": volume,
        "section_title": section,
        "chapter_num": chapter_num,
        "verse_range": verse_range,
        "refs_quran": refs,
        "hadith_refs": hadith,
        "label": label,
    }
