from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    SparseIndexParams,
    HnswConfigDiff,
)
from app.core.config import settings

COLLECTION_NAME = settings.QDRANT_COLLECTION
DENSE_DIM = 1024  # bge-m3 output dimension


def setup_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Create Qdrant collection with:
      - Dense vector (bge-m3, cosine)
      - Sparse vector (BM25)
      - Payload indexes for fast metadata filtering
    """
    exists = client.collection_exists(COLLECTION_NAME)

    if exists and not recreate:
        print(f"Collection '{COLLECTION_NAME}' already exists, skipping.")
        return

    if exists and recreate:
        print(f"Recreating collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

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

    # Payload indexes — chunk_id added for fast admin delete-by-document.
    for field, schema in [
        ("language", "keyword"),
        ("source", "keyword"),
        ("page_number", "integer"),
        ("is_ocr", "bool"),
        ("chunk_id", "keyword"),
        ("doc_id", "keyword"),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )

    print(f"Collection '{COLLECTION_NAME}' created with hybrid search enabled.")
