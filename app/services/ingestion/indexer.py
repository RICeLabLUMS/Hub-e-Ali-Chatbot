import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

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
