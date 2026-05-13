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
    ) -> list[dict]:
        query_emb = self.embedder.embed_query(query)

        # Treat 'unknown' as "no filter" — otherwise the retriever silently
        # returns zero results when the query language can't be pinned down.
        query_filter = None
        if language_filter and language_filter != "unknown":
            query_filter = Filter(must=[
                FieldCondition(key="language", match=MatchValue(value=language_filter))
            ])

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
                "score": r.score,
            }
            for r in results.points
        ]
