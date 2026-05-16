import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchExcept,
    MatchValue,
    PointStruct,
    SparseVector,
)

from app.services.ingestion.chunker import Chunk
from app.services.ingestion.embedder import BGEEmbedder
from app.services.qdrant_setup import COLLECTION_NAME

logger = logging.getLogger(__name__)

BATCH_SIZE = 50


class QdrantIndexer:
    """
    Embeds chunks and upserts them into Qdrant.

    Point IDs are deterministic (uuid5 of chunk_id), so re-ingesting the
    same PDF overwrites existing points rather than duplicating them.
    """

    def __init__(self, client: QdrantClient, embedder: BGEEmbedder):
        self.client = client
        self.embedder = embedder

    def index_chunks(self, chunks: list[Chunk], doc_id: str | None = None) -> int:
        if not chunks:
            return 0

        total = 0
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            self._index_batch(batch, doc_id)
            total += len(batch)
            logger.info(f"Indexed {total}/{len(chunks)} chunks")

        return total

    def index_chunks_replacing(
        self,
        chunks: list[Chunk],
        doc_id: str,
    ) -> int:
        """
        Atomic re-index for a single doc_id.

        Index-first, prune-stragglers pattern:
          1. Upsert all new chunks (deterministic uuid5(chunk_id) IDs overwrite
             matching old chunks in-place).
          2. After successful upsert, delete any chunks still tagged with this
             doc_id whose chunk_id is NOT in the new set (e.g. stale higher-
             indexed chunks left over from an edit that shortened the source).

        If step 1 fails, old chunks remain intact - no data loss. If step 2
        fails, we have orphan chunks (cosmetic) but new content is queryable.

        Caller passes the same `chunks` they'd pass to index_chunks; doc_id
        must be set so the stale-chunk filter can find leftovers.
        """
        if not chunks:
            # Empty new content: just clear all chunks for this doc_id.
            self._delete_by_doc_id(doc_id)
            return 0

        total = self.index_chunks(chunks, doc_id=doc_id)

        new_chunk_ids = [c.chunk_id for c in chunks]
        try:
            self._delete_stale_chunks(doc_id, keep_chunk_ids=new_chunk_ids)
        except Exception as e:
            logger.warning(
                f"Indexed {total} new chunks for {doc_id} but stale-chunk prune failed: {e}. "
                "Orphans may exist; not blocking - re-run --full-resync to clean up."
            )
        return total

    def _delete_stale_chunks(self, doc_id: str, keep_chunk_ids: list[str]) -> None:
        """Delete points with this doc_id whose chunk_id isn't in the keep list."""
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                        FieldCondition(key="chunk_id", match=MatchExcept(**{"except": keep_chunk_ids})),
                    ]
                )
            ),
            wait=True,
        )

    def _delete_by_doc_id(self, doc_id: str) -> None:
        """Delete every chunk with this doc_id. Used by index_chunks_replacing
        when the new content is empty (full wipe)."""
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ])
            ),
            wait=True,
        )

    def _index_batch(self, chunks: list[Chunk], doc_id: str | None) -> None:
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        points = []
        for chunk, emb in zip(chunks, embeddings):
            snippet = chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text
            points.append(PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                vector={
                    "dense": emb.dense,
                    "sparse": SparseVector(
                        indices=emb.sparse_indices,
                        values=emb.sparse_values,
                    ),
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": doc_id or chunk.chunk_id.split("__", 1)[0],
                    "text": chunk.text,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "language": chunk.language,
                    "is_ocr": chunk.is_ocr,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "token_count": chunk.token_count,
                    "snippet": snippet,
                    "title": chunk.title,
                    "url": chunk.url,
                    "content_type": chunk.content_type,
                },
            ))

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )
