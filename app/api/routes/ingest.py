import logging
import os
import uuid
from typing import Callable

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.config import settings
from app.core.dependencies import (
    get_embedder,
    get_job_queue,
    get_pdf_extractor,
    get_qdrant_client,
)
from app.core.security import verify_api_key
from app.services.ingestion.chunker import MultilingualChunker
from app.services.ingestion.indexer import QdrantIndexer
from app.services.ingestion.language_detector import detect_language

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    _: str = Depends(verify_api_key),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    doc_id = str(uuid.uuid4())[:8]
    upload_dir = settings.upload_dir_path
    tmp_path = str(upload_dir / f"{doc_id}_{file.filename}")

    async with aiofiles.open(tmp_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    queue = get_job_queue()
    await queue.submit(
        doc_id=doc_id,
        payload={"tmp_path": tmp_path, "filename": file.filename},
        fn=_process_pdf,
    )

    return {
        "message": "PDF queued for processing",
        "doc_id": doc_id,
        "filename": file.filename,
        "status_url": f"/api/ingest/status/{doc_id}",
    }


@router.get("/ingest/status/{doc_id}")
async def ingest_status(doc_id: str, _: str = Depends(verify_api_key)):
    status = get_job_queue().get_status(doc_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Unknown doc_id: {doc_id}")
    return {"doc_id": doc_id, **status}


@router.get("/ingest/status")
async def ingest_status_all(_: str = Depends(verify_api_key)):
    return get_job_queue().all_statuses()


async def _process_pdf(doc_id: str, payload: dict, status_cb: Callable[[dict], None]) -> None:
    tmp_path = payload["tmp_path"]
    filename = payload["filename"]

    try:
        status_cb({"message": "Extracting pages"})
        extractor = get_pdf_extractor()
        pages = extractor.extract(tmp_path)
        status_cb({"pages_extracted": len(pages)})
        logger.info(f"[{doc_id}] Extracted {len(pages)} pages from {filename}")

        # Tag each page with detected language (rough — chunker refines per segment)
        for page in pages:
            page.language = detect_language(page.text)

        status_cb({"message": "Chunking"})
        embedder = get_embedder()
        chunker = MultilingualChunker(embedding_model=embedder.dense_model)
        chunks = chunker.chunk_pages(pages, doc_id=doc_id)
        status_cb({"chunks_created": len(chunks)})
        logger.info(f"[{doc_id}] Created {len(chunks)} chunks")

        if not chunks:
            status_cb({"message": "No usable text extracted"})
            return

        status_cb({"message": "Embedding & indexing"})
        client = get_qdrant_client()
        indexer = QdrantIndexer(client=client, embedder=embedder)
        total = indexer.index_chunks(chunks, doc_id=doc_id)
        status_cb({"chunks_indexed": total})
        logger.info(f"[{doc_id}] Indexed {total} chunks")

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
