import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # pymupdf
import pymupdf4llm

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    text: str
    page_number: int
    source: str
    is_ocr: bool = False
    width: Optional[float] = None
    height: Optional[float] = None
    language: str = "unknown"   # populated by the ingest pipeline after extraction
    # Display metadata - propagated to chunks so the chat layer can render
    # nice citations ("<title> - <content_type>, p. <page>" with a clickable url).
    title: Optional[str] = None
    url: Optional[str] = None
    content_type: Optional[str] = None


class PDFExtractor:
    """
    Two-stage extractor:
      Stage 1: pymupdf4llm — fast, handles digital PDFs, RTL-aware
      Stage 2: surya-ocr   — fallback for scanned/image-only pages

    Surya models are loaded lazily and cached on the instance. Always
    obtain PDFExtractor via app.core.dependencies.get_pdf_extractor()
    so the cache survives across uploads.
    """

    MIN_TEXT_LENGTH = 50  # pages with less than this are treated as scanned

    def __init__(self) -> None:
        self._surya_models = None  # lazy loaded on first scanned page

    def extract(self, pdf_path: str) -> list[ExtractedPage]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        try:
            pages: list[ExtractedPage] = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                raw_text = page.get_text("text").strip()

                if len(raw_text) >= self.MIN_TEXT_LENGTH:
                    md_text = self._extract_with_pymupdf4llm(pdf_path, page_num)
                    pages.append(ExtractedPage(
                        text=md_text or raw_text,
                        page_number=page_num + 1,
                        source=path.name,
                        is_ocr=False,
                        width=page.rect.width,
                        height=page.rect.height,
                    ))
                else:
                    logger.info(f"Page {page_num + 1} appears scanned, using OCR")
                    ocr_text = self._extract_with_surya(page)
                    pages.append(ExtractedPage(
                        text=ocr_text,
                        page_number=page_num + 1,
                        source=path.name,
                        is_ocr=True,
                        width=page.rect.width,
                        height=page.rect.height,
                    ))
        finally:
            doc.close()

        return [p for p in pages if p.text.strip()]

    def _extract_with_pymupdf4llm(self, pdf_path: str, page_num: int) -> str:
        try:
            md = pymupdf4llm.to_markdown(
                pdf_path,
                pages=[page_num],
                show_progress=False,
            )
            return md.strip()
        except Exception as e:
            logger.warning(f"pymupdf4llm failed on page {page_num}: {e}")
            doc = fitz.open(pdf_path)
            try:
                return doc[page_num].get_text("text").strip()
            finally:
                doc.close()

    def _get_surya_models(self):
        if self._surya_models is not None:
            return self._surya_models
        try:
            from surya.model.detection.model import load_model as load_det_model
            from surya.model.detection.processor import load_processor as load_det_processor
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
        except ImportError:
            logger.error("surya-ocr not installed. Install with: pip install surya-ocr")
            self._surya_models = False  # sentinel so we don't retry the import per page
            return False

        det_model = load_det_model()
        det_processor = load_det_processor()
        rec_model = load_rec_model()
        rec_processor = load_rec_processor()
        self._surya_models = (det_model, det_processor, rec_model, rec_processor)
        return self._surya_models

    def _extract_with_surya(self, page: fitz.Page) -> str:
        models = self._get_surya_models()
        if not models:
            return ""

        try:
            import numpy as np
            from surya.ocr import run_ocr

            det_model, det_processor, rec_model, rec_processor = models

            # Render page to image at 2x for OCR accuracy
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            results = run_ocr(
                [img_array],
                [["en", "ar", "ur"]],
                det_model,
                det_processor,
                rec_model,
                rec_processor,
            )

            lines = [
                block.text for block in results[0].text_lines
                if block.confidence > 0.6
            ]
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"surya OCR failed: {e}", exc_info=True)
            return ""
