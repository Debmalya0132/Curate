"""
FAISS vector index for fast nearest-neighbor search over product embeddings.

Uses IndexFlatIP (inner product) which is equivalent to cosine similarity
when all vectors are L2-normalized.
"""

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorIndex:
    """FAISS-backed similarity search index with persistence."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._id_map: list[str] = []
        logger.info("VectorIndex created (dim=%d)", dim)

    # ── Properties ─────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self.index.ntotal

    # ── Core ops ───────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, ids: list[str]) -> None:
        """Add L2-normalized embeddings with associated product IDs."""
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} ≠ index dim {self.dim}"
            )
        if embeddings.shape[0] != len(ids):
            raise ValueError(
                f"{embeddings.shape[0]} embeddings but {len(ids)} IDs"
            )

        self.index.add(embeddings)
        self._id_map.extend(ids)
        logger.info("Added %d vectors → total %d", len(ids), self.size)

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k (product_id, similarity_score) pairs."""
        if self.size == 0:
            logger.warning("Search on empty index")
            return []

        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        k = min(k, self.size)

        scores, indices = self.index.search(query, k)
        return [
            (self._id_map[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    # ── Persistence ────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """Write index + id map to disk."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "faiss.index"))
        np.save(str(d / "id_map.npy"), np.array(self._id_map))
        logger.info("Saved index (%d vectors) → %s", self.size, d)

    def load(self, directory: str | Path) -> None:
        """Load a previously saved index from disk."""
        d = Path(directory)
        idx_path = d / "faiss.index"
        ids_path = d / "id_map.npy"
        if not idx_path.exists():
            raise FileNotFoundError(f"No index at {idx_path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"No id map at {ids_path}")

        self.index = faiss.read_index(str(idx_path))
        self._id_map = list(np.load(str(ids_path), allow_pickle=True))
        self.dim = self.index.d
        logger.info("Loaded index (%d vectors) ← %s", self.size, d)

    def reset(self) -> None:
        """Clear the index."""
        self.index = faiss.IndexFlatIP(self.dim)
        self._id_map.clear()
        logger.info("Index reset")
