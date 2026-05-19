import logging
import re
from dataclasses import dataclass
from typing import Optional

from chonkie import SemanticChunker
from chonkie.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer

from app.services.ingestion.citation_extractor import (
    extract_hadith_refs,
    extract_quran_refs,
    extract_section_title,
)
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
    # Display metadata (copied from ExtractedPage on chunking).
    title: Optional[str] = None
    url: Optional[str] = None
    content_type: Optional[str] = None
    # Numeric citations - some inherited from the page (doc-level), some
    # extracted per chunk after the chunker has split the text.
    chapter_num: Optional[int] = None        # doc-level (inherited)
    verse_range: Optional[str] = None        # doc-level (inherited)
    volume: Optional[int] = None             # doc-level (inherited)
    refs_quran: Optional[list[str]] = None   # chunk-level: ["8:1", "8:5-8"]
    section_title: Optional[str] = None      # chunk-level: "VERSE 1" / "Sermon 17"
    hadith_refs: Optional[list[str]] = None  # chunk-level: ["Vol. 1 H. 245", "Sermon 17"]


class MultilingualChunker:
    """
    Semantic chunker over bge-m3 sentence embeddings.

    Mixed-language pages are split paragraph-wise by language *before*
    semantic chunking, so each chunk has a single, accurate language tag.
    """

    # bge-m3's max sequence length is 8192 tokens. We pre-split above this so
    # chonkie's internal tokenization never trips the "10240 > 8192" warning
    # from the underlying HF tokenizer. 4000 tokens leaves plenty of headroom
    # for semantic boundary detection inside each piece.
    MAX_TOKENS_PER_SEGMENT = 4000
    # Fallback char window when no tokenizer is available - intentionally
    # conservative for non-Latin scripts (Arabic/Urdu pack ~1.6 chars/token).
    CHAR_FALLBACK_PER_SEGMENT = 6000

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
            self._tokenizer = getattr(embedding_model, "tokenizer", None)
        else:
            wrapped = embedding_model
            self._tokenizer = None

        # Pre-set the once-only "sequence too long" warning flag so our own
        # token-counting calls don't trip it. chonkie's later calls also stay
        # quiet because every sub-segment we pass in fits under model_max_length.
        if self._tokenizer is not None and hasattr(self._tokenizer, "deprecation_warnings"):
            self._tokenizer.deprecation_warnings[
                "sequence-length-is-longer-than-the-specified-maximum"
            ] = True

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
                for sub_idx, sub_segment in enumerate(
                    self._split_long_segment(segment_text)
                ):
                    try:
                        raw_chunks = self.chunker.chunk(sub_segment)
                    except Exception as e:
                        logger.warning(
                            f"Chunker failed on page {page.page_number} segment {seg_idx}/{sub_idx}: {e}; "
                            f"falling back to whole sub-segment"
                        )
                        raw_chunks = [_PseudoChunk(sub_segment)]

                    for i, raw in enumerate(raw_chunks):
                        chunk_text = raw.text.strip()
                        if len(chunk_text) < self.min_chunk_chars:
                            dropped += 1
                            continue

                        # Per-chunk numeric citations - run AFTER chunking
                        # so each chunk gets refs/section relevant to its
                        # own slice of the page text.
                        refs = extract_quran_refs(chunk_text) or None
                        section = extract_section_title(chunk_text)
                        hadith = extract_hadith_refs(chunk_text) or None

                        chunk = Chunk(
                            text=chunk_text,
                            chunk_id=f"{doc_id}__p{page.page_number}__s{seg_idx}_{sub_idx}__c{i}",
                            source=page.source,
                            page_number=page.page_number,
                            language=seg_lang,
                            is_ocr=page.is_ocr,
                            char_start=getattr(raw, "start_index", 0),
                            char_end=getattr(raw, "end_index", len(chunk_text)),
                            token_count=getattr(raw, "token_count", None),
                            title=getattr(page, "title", None),
                            url=getattr(page, "url", None),
                            content_type=getattr(page, "content_type", None),
                            # Doc-level fields inherited from the page.
                            chapter_num=getattr(page, "chapter_num", None),
                            verse_range=getattr(page, "verse_range", None),
                            volume=getattr(page, "volume", None),
                            # Chunk-level fields extracted just now.
                            refs_quran=refs,
                            section_title=section,
                            hadith_refs=hadith,
                        )
                        all_chunks.append(chunk)

        if dropped:
            logger.debug(f"Dropped {dropped} chunks below {self.min_chunk_chars} chars")
        return all_chunks

    def _split_long_segment(self, text: str) -> list[str]:
        """
        Split text into pieces each within the model's token budget. When the
        tokenizer is available we count tokens precisely (necessary because
        Arabic/Urdu pack many more tokens per character than English). Falls
        back to a conservative character window if no tokenizer.
        """
        if self._tokenizer is None:
            return self._split_by_chars(text, self.CHAR_FALLBACK_PER_SEGMENT)
        return self._split_by_tokens(text, self.MAX_TOKENS_PER_SEGMENT)

    def _split_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        # Quick token count - if under budget, nothing to split.
        total_ids = self._tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if len(total_ids) <= max_tokens:
            return [text]

        paragraphs = re.split(r"\n\s*\n", text)
        pieces: list[str] = []
        buf: list[str] = []
        buf_tokens = 0

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            p_ids = self._tokenizer.encode(p, add_special_tokens=False, truncation=False)
            p_tokens = len(p_ids)

            # Single paragraph too big: slice its token IDs into windows.
            if p_tokens > max_tokens:
                if buf:
                    pieces.append("\n\n".join(buf))
                    buf, buf_tokens = [], 0
                for i in range(0, p_tokens, max_tokens):
                    sub_ids = p_ids[i : i + max_tokens]
                    pieces.append(self._tokenizer.decode(sub_ids, skip_special_tokens=True))
                continue

            if buf_tokens + p_tokens > max_tokens and buf:
                pieces.append("\n\n".join(buf))
                buf, buf_tokens = [p], p_tokens
            else:
                buf.append(p)
                buf_tokens += p_tokens

        if buf:
            pieces.append("\n\n".join(buf))
        return pieces

    @staticmethod
    def _split_by_chars(text: str, max_chars: int) -> list[str]:
        """Fallback char-based splitter when no tokenizer is available."""
        if len(text) <= max_chars:
            return [text]
        paragraphs = re.split(r"\n\s*\n", text)
        pieces: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(p) > max_chars:
                if buf:
                    pieces.append("\n\n".join(buf))
                    buf, buf_len = [], 0
                for i in range(0, len(p), max_chars):
                    pieces.append(p[i : i + max_chars])
                continue
            if buf_len + len(p) + 2 > max_chars and buf:
                pieces.append("\n\n".join(buf))
                buf, buf_len = [p], len(p)
            else:
                buf.append(p)
                buf_len += len(p) + 2
        if buf:
            pieces.append("\n\n".join(buf))
        return pieces

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
