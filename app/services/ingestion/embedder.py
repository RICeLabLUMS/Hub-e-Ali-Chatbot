import logging
from dataclasses import dataclass

from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    dense: list[float]          # 1024-dim for bge-m3
    sparse_indices: list[int]
    sparse_values: list[float]


class BGEEmbedder:
    """Dense (bge-m3) + sparse (BM25) embedder, lazy-loaded once."""

    DENSE_MODEL = "BAAI/bge-m3"
    SPARSE_MODEL = "Qdrant/bm25"

    def __init__(self) -> None:
        device = self._resolve_device(settings.EMBEDDING_DEVICE)
        logger.info(f"Loading bge-m3 dense model on {device}...")
        self.dense_model = SentenceTransformer(self.DENSE_MODEL, device=device)

        logger.info("Loading BM25 sparse model...")
        self.sparse_model = SparseTextEmbedding(model_name=self.SPARSE_MODEL)

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        dense_vecs = self.dense_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sparse_results = list(self.sparse_model.embed(texts))

        return [
            EmbeddingResult(
                dense=dense_vecs[i].tolist(),
                sparse_indices=sparse_results[i].indices.tolist(),
                sparse_values=sparse_results[i].values.tolist(),
            )
            for i in range(len(texts))
        ]

    def embed_query(self, query: str) -> EmbeddingResult:
        instructed = f"Represent this sentence for searching relevant passages: {query}"
        dense_vec = self.dense_model.encode(
            [instructed],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        sparse = list(self.sparse_model.embed([query]))[0]

        return EmbeddingResult(
            dense=dense_vec.tolist(),
            sparse_indices=sparse.indices.tolist(),
            sparse_values=sparse.values.tolist(),
        )

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
        # auto
        return "cuda" if cuda else "cpu"
