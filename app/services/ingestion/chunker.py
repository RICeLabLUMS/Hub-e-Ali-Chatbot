import logging
import re
from dataclasses import dataclass
from typing import Optional

from chonkie import SemanticChunker
from chonkie.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer

from app.services.ingestion.language_detector import split_by_language

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    chunk_id: str           # e.g. "doc_abc__p3__s0__c2"
    source: str
    page_number: int
    language: str
    is_ocr: bool
    char_start: int
    char_end: int
    token_count: Optional[int] = None


class MultilingualChunker:
    """
    Semantic chunker over bge-m3 sentence embeddings.

    Mixed-language pages are split paragraph-wise by language *before*
    semantic chunking, so each chunk has a single, accurate language tag.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        chunk_size: int = 512,
        threshold: float = 0.5,
        min_chunk_chars: int = 30,
    ):
        # chonkie's SemanticChunker wants a string or a chonkie BaseEmbeddings,
        # not a raw SentenceTransformer. Wrap the already-loaded model so we
        # don't pay the bge-m3 download/load twice.
        if isinstance(embedding_model, SentenceTransformer):
            wrapped = SentenceTransformerEmbeddings(model=embedding_model)
        else:
            wrapped = embedding_model

        self.chunker = SemanticChunker(
            embedding_model=wrapped,
            chunk_size=chunk_size,
            threshold=threshold,
        )
        self.min_chunk_chars = min_chunk_chars

    def chunk_pages(self, pages: list, doc_id: str) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        dropped = 0

        for page in pages:
            text = self._clean_text(page.text)
            if not text:
                continue

            segments = split_by_language(text)
            if not segments:
                continue

            for seg_idx, (segment_text, seg_lang) in enumerate(segments):
                try:
                    raw_chunks = self.chunker.chunk(segment_text)
                except Exception as e:
                    logger.warning(
                        f"Chunker failed on page {page.page_number} segment {seg_idx}: {e}; "
                        f"falling back to whole segment"
                    )
                    raw_chunks = [_PseudoChunk(segment_text)]

                for i, raw in enumerate(raw_chunks):
                    chunk_text = raw.text.strip()
                    if len(chunk_text) < self.min_chunk_chars:
                        dropped += 1
                        continue

                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=f"{doc_id}__p{page.page_number}__s{seg_idx}__c{i}",
                        source=page.source,
                        page_number=page.page_number,
                        language=seg_lang,
                        is_ocr=page.is_ocr,
                        char_start=getattr(raw, "start_index", 0),
                        char_end=getattr(raw, "end_index", len(chunk_text)),
                        token_count=getattr(raw, "token_count", None),
                    )
                    all_chunks.append(chunk)

        if dropped:
            logger.debug(f"Dropped {dropped} chunks below {self.min_chunk_chars} chars")
        return all_chunks

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"[​‌‍﻿]", "", text)
        return text.strip()


@dataclass
class _PseudoChunk:
    """Fallback when chonkie fails on a segment — treat the whole segment as one chunk."""
    text: str
    start_index: int = 0
    end_index: int = 0
    token_count: Optional[int] = None

    def __post_init__(self):
        self.end_index = len(self.text)
