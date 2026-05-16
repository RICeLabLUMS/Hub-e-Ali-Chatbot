import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # pymupdf

logger = logging.getLogger(__name__)

# Silence PyMuPDF's C-level stderr banners ("=== Document parser messages ===").
# We control OCR explicitly via Surya in Stage 2 and don't want noise from the
# underlying mupdf library. Use False to suppress; True to re-enable for debugging.
try:
    fitz.TOOLS.mupdf_display_errors(False)
except AttributeError:
    pass


def _detect_columns(centroids: list[float], page_width_px: int) -> list[tuple[float, float]]:
    """
    Heuristic two-column detection. Looks for a single large horizontal gap
    in the x-centroid distribution that spans > 15% of page width and falls
    near the middle. If found, splits into two columns; else single column.

    Robust enough for typical book pages without bringing in surya_layout.
    Multi-column (3+) pages are uncommon in this corpus; would be misordered
    but text is still captured.
    """
    if len(centroids) < 4:
        return [(0.0, float(page_width_px))]

    sorted_cs = sorted(centroids)
    # Find the largest contiguous gap in sorted centroids
    max_gap = 0.0
    gap_left = gap_right = 0.0
    for i in range(len(sorted_cs) - 1):
        g = sorted_cs[i + 1] - sorted_cs[i]
        if g > max_gap:
            max_gap = g
            gap_left = sorted_cs[i]
            gap_right = sorted_cs[i + 1]

    # Require the gap to be meaningfully wide AND straddle the page midline
    midpoint_zone = (page_width_px * 0.30, page_width_px * 0.70)
    boundary = (gap_left + gap_right) / 2
    if (
        max_gap > page_width_px * 0.15
        and midpoint_zone[0] <= boundary <= midpoint_zone[1]
    ):
        return [(0.0, boundary), (boundary, float(page_width_px))]
    return [(0.0, float(page_width_px))]


def _group_into_paragraphs(lines: list) -> str:
    """
    Group vertically-adjacent lines into paragraphs.

    A paragraph break is declared when the vertical gap between two consecutive
    lines is more than ~1.7× the median line height in the current group -
    that's the typical inter-paragraph leading in printed books, regardless of
    font size or script. Within a paragraph, lines are joined with a space
    (collapsing soft line breaks from justified text).
    """
    if not lines:
        return ""

    heights = sorted((tl.bbox[3] - tl.bbox[1]) for tl in lines)
    median_h = heights[len(heights) // 2] if heights else 20.0
    paragraph_gap_threshold = median_h * 1.7

    paragraphs: list[str] = []
    current: list[str] = [lines[0].text.strip()]
    prev_bottom = lines[0].bbox[3]

    for tl in lines[1:]:
        gap = tl.bbox[1] - prev_bottom
        text = (tl.text or "").strip()
        if not text:
            prev_bottom = tl.bbox[3]
            continue
        if gap > paragraph_gap_threshold:
            paragraphs.append(" ".join(current))
            current = [text]
        else:
            current.append(text)
        prev_bottom = tl.bbox[3]

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(p for p in paragraphs if p)


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
        self._surya_load_failed = False  # latched after first failed load

    def extract(self, pdf_path: str) -> list[ExtractedPage]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        ocr_attempted = 0
        ocr_empty = 0
        try:
            pages: list[ExtractedPage] = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                # page.get_text("text") only reads the PDF's text layer - no OCR.
                # Earlier versions of this code routed through pymupdf4llm.to_markdown
                # which internally invokes PyMuPDF's Tesseract OCR path for image
                # regions, producing low-quality Arabic/Urdu output. Stage 2 (Surya)
                # is the only OCR path now, and only runs on pages with no text layer.
                raw_text = page.get_text("text").strip()

                if len(raw_text) >= self.MIN_TEXT_LENGTH:
                    pages.append(ExtractedPage(
                        text=raw_text,
                        page_number=page_num + 1,
                        source=path.name,
                        is_ocr=False,
                        width=page.rect.width,
                        height=page.rect.height,
                    ))
                else:
                    logger.info(f"Page {page_num + 1} appears scanned, using OCR")
                    ocr_attempted += 1
                    # _extract_with_surya raises RuntimeError if surya itself
                    # cannot be loaded (fail-fast, not silent). For per-page
                    # recognition failures it returns "" so one bad page doesn't
                    # nuke a whole book; we count and report those below.
                    ocr_text = self._extract_with_surya(page)
                    if not ocr_text.strip():
                        ocr_empty += 1
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

        if ocr_empty:
            logger.error(
                f"OCR produced empty text on {ocr_empty}/{ocr_attempted} scanned page(s) "
                f"of {path.name} - those pages will be dropped from the index. "
                "Check earlier surya errors for the cause."
            )
        return [p for p in pages if p.text.strip()]

    def _get_surya_models(self):
        """
        Lazy-load Surya predictors and cache them on the instance.

        Surya rewrote its API around v0.6 - the old surya.model.* paths were
        removed. We use the new Predictor classes (DetectionPredictor /
        RecognitionPredictor), which encapsulate model + processor and accept
        an optional device argument.

        Returns: (det_predictor, rec_predictor)
        Raises RuntimeError on any unrecoverable load failure (loud, not silent).
        """
        # Already loaded - reuse.
        if self._surya_models is not None and self._surya_models is not False:
            return self._surya_models
        # Latched failure - don't keep retrying noisy imports per page, but
        # also don't silently produce empty pages.
        if self._surya_load_failed:
            raise RuntimeError(
                "Surya OCR is unavailable (load previously failed). "
                "Scanned pages cannot be processed. "
                "Install/repair with: pip install surya-ocr"
            )
        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
        except ImportError as e:
            self._surya_load_failed = True
            logger.error(
                "Surya OCR import failed. Either surya-ocr is not installed "
                "(pip install surya-ocr) or its API has changed again. "
                f"Underlying error: {e}"
            )
            raise RuntimeError(
                "Surya OCR import failed. See logs for details. "
                "Install with: pip install surya-ocr"
            ) from e

        try:
            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor()
        except Exception as e:
            self._surya_load_failed = True
            logger.error(f"surya model load failed: {e}", exc_info=True)
            raise RuntimeError(f"Surya OCR model load failed: {e}") from e

        self._surya_models = (det_predictor, rec_predictor)
        return self._surya_models

    def _extract_with_surya(self, page: fitz.Page) -> str:
        # Raises RuntimeError if Surya cannot be loaded - we want the caller
        # (and ultimately the sync report) to mark this PDF as failed rather
        # than silently producing zero chunks.
        det_predictor, rec_predictor = self._get_surya_models()

        try:
            import numpy as np
            from PIL import Image
            from app.core.config import settings

            # Render at configured scale (default 3x). PyMuPDF returns RGB(A)
            # pixels; surya is happy with grayscale, which also helps the
            # recognizer focus on glyphs rather than background color noise.
            scale = float(settings.OCR_RENDER_SCALE or 3.0)
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            # Convert to grayscale PIL image - small accuracy boost + cheaper.
            if pix.n >= 3:
                pil_img = Image.fromarray(img_array[:, :, :3]).convert("L")
            else:
                pil_img = Image.fromarray(img_array.squeeze())

            # Surya >=0.6 API: rec_predictor is callable, takes images +
            # langs (one list per image) + the det_predictor it should chain
            # text-line detection through. Returns a list of OCRResult.
            results = rec_predictor(
                [pil_img],
                [["en", "ar", "ur"]],
                det_predictor,
            )

            return self._reconstruct_reading_order(
                results[0].text_lines,
                page_width_px=pil_img.width,
                confidence_threshold=float(settings.OCR_CONFIDENCE_THRESHOLD or 0.6),
            )

        except Exception as e:
            # Per-page recognition failure (model loaded but page-level issue):
            # log loudly with the page number context but return "" so a single
            # bad page doesn't nuke the whole PDF. extract() counts these and
            # surfaces a summary at the end.
            logger.error(
                f"surya OCR failed on page {page.number + 1}: {e}",
                exc_info=True,
            )
            return ""

    @staticmethod
    def _reconstruct_reading_order(
        text_lines,
        page_width_px: int,
        confidence_threshold: float,
    ) -> str:
        """
        Turn Surya line outputs into natural reading order, preserving paragraph
        structure. Surya gives us each line's bbox + confidence; we use those to:
          1. drop low-confidence lines (noise pollutes embeddings)
          2. detect single vs two-column layout via x-centroid clustering
          3. sort each column top-to-bottom
          4. group adjacent lines into paragraphs by vertical gap vs line height

        Output: paragraphs separated by '\n\n', columns by '\n\n' too. The
        downstream chunker treats blank lines as paragraph boundaries, so this
        gives semantically clean chunks instead of one chunk per line.
        """
        if not text_lines:
            return ""

        # 1. Confidence filter
        confident = [
            tl for tl in text_lines
            if (getattr(tl, "confidence", None) or 0) >= confidence_threshold
            and (tl.text or "").strip()
        ]
        if not confident:
            return ""

        # 2. Detect columns by looking for a big gap in x-centroid distribution.
        centroids = [(tl.bbox[0] + tl.bbox[2]) / 2 for tl in confident]
        column_ranges = _detect_columns(centroids, page_width_px)

        # 3+4. Process each column.
        column_texts: list[str] = []
        for col_x0, col_x1 in column_ranges:
            in_col = [
                tl for tl in confident
                if col_x0 <= ((tl.bbox[0] + tl.bbox[2]) / 2) < col_x1
            ]
            in_col.sort(key=lambda tl: tl.bbox[1])  # top to bottom
            column_texts.append(_group_into_paragraphs(in_col))

        # Reading order across columns: left-to-right for mixed-language /
        # English; users with pure-RTL multi-column books may want reversed
        # order, but that's rare in this corpus and detect_language can't help
        # us pre-OCR. Keeping LTR for now.
        return "\n\n".join(t for t in column_texts if t.strip())
