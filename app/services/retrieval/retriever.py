import asyncio

from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.services.ingestion.embedder import BGEEmbedder
from app.services.qdrant_setup import COLLECTION_NAME


class HybridRetriever:
    """Dense + sparse hybrid retrieval with RRF fusion in Qdrant."""

    def __init__(self, client: QdrantClient, embedder: BGEEmbedder):
        self.client = client
        self.embedder = embedder

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        language_filter: str | None = None,
        content_type_filter: str | list[str] | None = None,
    ) -> list[dict]:
        query_emb = self.embedder.embed_query(query)

        # Treat 'unknown'/empty as "no filter" - otherwise the retriever
        # silently returns zero results when the query language can't be
        # pinned down or when the UI passes a vacuous 'all' source filter.
        must: list = []
        if language_filter and language_filter != "unknown":
            must.append(FieldCondition(
                key="language", match=MatchValue(value=language_filter)
            ))
        if content_type_filter:
            if isinstance(content_type_filter, str):
                must.append(FieldCondition(
                    key="content_type", match=MatchValue(value=content_type_filter)
                ))
            else:
                # Multi-value filter via OR'd nested filter inside `must`.
                must.append(Filter(should=[
                    FieldCondition(key="content_type", match=MatchValue(value=v))
                    for v in content_type_filter
                ]))
        query_filter = Filter(must=must) if must else None

        # Qdrant client is sync — run in a thread so we don't block the event loop.
        def _query():
            return self.client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    Prefetch(
                        query=query_emb.dense,
                        using="dense",
                        limit=top_k * 2,
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=query_emb.sparse_indices,
                            values=query_emb.sparse_values,
                        ),
                        using="sparse",
                        limit=top_k * 2,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

        results = await asyncio.to_thread(_query)

        return [
            {
                "chunk_id": r.payload.get("chunk_id"),
                "text": r.payload.get("text", ""),
                "source": r.payload.get("source", ""),
                "page": r.payload.get("page_number"),
                "lang": r.payload.get("language"),
                "snippet": r.payload.get("snippet", ""),
                "is_ocr": r.payload.get("is_ocr", False),
                "title": r.payload.get("title"),
                "url": r.payload.get("url"),
                "content_type": r.payload.get("content_type"),
                # Numeric citations
                "chapter_num": r.payload.get("chapter_num"),
                "verse_range": r.payload.get("verse_range"),
                "volume": r.payload.get("volume"),
                "refs_quran": r.payload.get("refs_quran") or [],
                "section_title": r.payload.get("section_title"),
                "hadith_refs": r.payload.get("hadith_refs") or [],
                "score": r.score,
            }
            for r in results.points
        ]
