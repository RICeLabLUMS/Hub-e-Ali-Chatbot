"""
Bulk PDF loader.

Two modes:
  --via-api   (default): POST each PDF to /api/ingest/pdf (server must be running)
  --offline           : drive the ingestion pipeline directly, no HTTP

Examples:
  python scripts/load_documents.py
  python scripts/load_documents.py --offline
  python scripts/load_documents.py --folder data --api http://localhost:8000
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import httpx

# Make `app.*` importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("loader")


async def upload_via_api(folder: Path, api_url: str, api_key: str) -> None:
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDFs found in {folder}")
        return

    async with httpx.AsyncClient(timeout=300.0) as client:
        for pdf in pdfs:
            logger.info(f"Uploading {pdf.name}...")
            with pdf.open("rb") as f:
                resp = await client.post(
                    f"{api_url}/api/ingest/pdf",
                    headers={"X-API-Key": api_key},
                    files={"file": (pdf.name, f, "application/pdf")},
                )
            resp.raise_for_status()
            data = resp.json()
            doc_id = data["doc_id"]
            logger.info(f"  queued as doc_id={doc_id}")

            # Poll status
            while True:
                await asyncio.sleep(3)
                s = await client.get(
                    f"{api_url}/api/ingest/status/{doc_id}",
                    headers={"X-API-Key": api_key},
                )
                if s.status_code != 200:
                    logger.error(f"  status check failed: {s.text}")
                    break
                state = s.json().get("state")
                logger.info(f"  {pdf.name}: {state}")
                if state in ("done", "failed"):
                    if state == "failed":
                        logger.error(f"  error: {s.json().get('error')}")
                    break


async def load_offline(folder: Path) -> None:
    """Run the full ingestion pipeline in-process — no HTTP server needed."""
    from app.core.dependencies import get_embedder, get_pdf_extractor, get_qdrant_client
    from app.services.ingestion.chunker import MultilingualChunker
    from app.services.ingestion.indexer import QdrantIndexer
    from app.services.ingestion.language_detector import detect_language
    from app.services.qdrant_setup import setup_collection

    setup_collection(get_qdrant_client(), recreate=False)

    extractor = get_pdf_extractor()
    embedder = get_embedder()
    chunker = MultilingualChunker(embedding_model=embedder.dense_model)
    indexer = QdrantIndexer(client=get_qdrant_client(), embedder=embedder)

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDFs found in {folder}")
        return

    for pdf in pdfs:
        doc_id = pdf.stem.replace(" ", "_")[:32]
        logger.info(f"Processing {pdf.name} as doc_id={doc_id}...")

        pages = extractor.extract(str(pdf))
        # Display metadata: local files have no canonical URL, so url stays None.
        for p in pages:
            p.language = detect_language(p.text)
            p.title = pdf.stem
            p.content_type = "PDF"

        chunks = chunker.chunk_pages(pages, doc_id=doc_id)
        total = indexer.index_chunks(chunks, doc_id=doc_id)
        logger.info(f"  indexed {total} chunks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data", help="Folder containing PDFs")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "change-me-in-production"))
    parser.add_argument("--offline", action="store_true", help="Run ingestion in-process")
    args = parser.parse_args()

    folder = Path(args.folder)
    folder.mkdir(exist_ok=True)

    if args.offline:
        asyncio.run(load_offline(folder))
    else:
        asyncio.run(upload_via_api(folder, args.api, args.api_key))


if __name__ == "__main__":
    main()
