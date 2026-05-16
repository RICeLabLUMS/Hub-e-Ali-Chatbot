import logging

from sentence_transformers import CrossEncoder

from app.core.config import settings

logger = logging.getLogger(__name__)


class BGEReranker:
    """bge-reranker-v2-m3 — multilingual cross-encoder."""

    MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self) -> None:
        device = self._resolve_device(settings.EMBEDDING_DEVICE)
        max_length = int(settings.RERANKER_MAX_LENGTH or 1024)
        logger.info(f"Loading bge-reranker-v2-m3 on {device} (max_length={max_length})...")
        self.model = CrossEncoder(self.MODEL, device=device, max_length=max_length)

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
        if not chunks:
            return []
        if top_k is None:
            top_k = settings.RERANK_TOP_K

        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _resolve_device(pref: str) -> str:
        pref = (pref or "auto").lower()
        if pref == "cpu":
            return "cpu"
        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:
            cuda = False
        if pref == "cuda":
            return "cuda" if cuda else "cpu"
        return "cuda" if cuda else "cpu"
