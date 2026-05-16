import logging

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    SparseIndexParams,
    HnswConfigDiff,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.QDRANT_COLLECTION
DENSE_DIM = 1024  # bge-m3 output dimension

PAYLOAD_INDEXES: list[tuple[str, str]] = [
    ("language", "keyword"),
    ("source", "keyword"),
    ("page_number", "integer"),
    ("is_ocr", "bool"),
    ("chunk_id", "keyword"),
    ("doc_id", "keyword"),
    ("content_type", "keyword"),  # enables UI filters by Article/Page/PDF/etc.
]


def setup_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Create Qdrant collection with:
      - Dense vector (bge-m3, cosine)
      - Sparse vector (BM25)
      - Payload indexes for fast metadata filtering

    Payload indexes are created idempotently on every call so newly added
    fields propagate to existing collections without requiring a recreate.
    """
    exists = client.collection_exists(COLLECTION_NAME)

    if exists and recreate:
        print(f"Recreating collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        print(f"Collection '{COLLECTION_NAME}' created with hybrid search enabled.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists, ensuring payload indexes.")

    _ensure_payload_indexes(client)


def _ensure_payload_indexes(client: QdrantClient) -> None:
    for field, schema in PAYLOAD_INDEXES:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
            )
        except (UnexpectedResponse, ValueError) as e:
            # qdrant returns 409 / "already exists" when the index is already there.
            msg = str(e).lower()
            if "already" in msg or "exists" in msg or "duplicate" in msg:
                continue
            logger.warning(f"Payload index create failed for {field!r}: {e}")
