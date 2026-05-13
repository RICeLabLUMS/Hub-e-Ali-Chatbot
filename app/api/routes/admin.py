from fastapi import APIRouter, Depends, HTTPException

from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

from app.core.dependencies import get_qdrant_client
from app.core.security import verify_api_key
from app.services.qdrant_setup import COLLECTION_NAME, setup_collection

router = APIRouter()


@router.get("/admin/collection/stats")
async def collection_stats(_: str = Depends(verify_api_key)):
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        raise HTTPException(status_code=404, detail="Collection does not exist")
    info = client.get_collection(COLLECTION_NAME)
    count = client.count(COLLECTION_NAME, exact=True).count
    return {
        "collection": COLLECTION_NAME,
        "point_count": count,
        "status": str(info.status),
        "vectors_count": info.vectors_count,
        "segments_count": info.segments_count,
    }


@router.post("/admin/collection/recreate")
async def collection_recreate(_: str = Depends(verify_api_key)):
    client = get_qdrant_client()
    setup_collection(client, recreate=True)
    return {"message": f"Collection '{COLLECTION_NAME}' recreated"}


@router.delete("/admin/document/{doc_id}")
async def delete_document(doc_id: str, _: str = Depends(verify_api_key)):
    client = get_qdrant_client()
    result = client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            ])
        ),
        wait=True,
    )
    return {"message": f"Deleted points for doc_id={doc_id}", "operation_id": str(result.operation_id)}
