"""
embedding_service.py — Local semantic embedding using Snowflake Arctic Embed v2.

The model runs entirely on-premise to ensure no data leaves the network.
Embeddings are L2-normalised before storage / comparison.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

import config


class EmbeddingService:
    """Singleton wrapper around the Snowflake Arctic local embedding model."""

    _instance: EmbeddingService | None = None

    def __new__(cls) -> EmbeddingService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        print(
            f"[EmbeddingService] Loading model: {config.EMBEDDING_MODEL_NAME} …"
        )
        self._model = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device="cpu",
        )
        # Store prompt prefix for multilingual-e5 models (improves retrieval)
        model_lower = config.EMBEDDING_MODEL_NAME.lower()
        self._query_prefix = "query: " if "e5" in model_lower else ""
        self._passage_prefix = "passage: " if "e5" in model_lower else ""
        self._initialised = True
        print("[EmbeddingService] Model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(self, text: str, is_query: bool = True) -> np.ndarray:
        """
        Return a normalised 1-D float32 embedding for *text*.

        For multilingual-e5 models, prepends the appropriate prefix
        ("query: " for queries, "passage: " for stored documents).

        Args:
            text:     Input string (Hebrew or English, any length).
            is_query: True when encoding a search query; False for a passage.

        Returns:
            np.ndarray of shape (D,), dtype float32, L2-norm ≈ 1.0.
        """
        prefix = self._query_prefix if is_query else self._passage_prefix
        vec: np.ndarray = self._model.encode(
            prefix + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def encode_batch(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """
        Batch-encode a list of strings.

        Returns:
            np.ndarray of shape (N, D), dtype float32.
        """
        prefix = self._query_prefix if is_query else self._passage_prefix
        prefixed = [prefix + t for t in texts]
        vecs: np.ndarray = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return vecs.astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two normalised vectors.

        Because both are L2-normalised, this is simply their dot product.
        """
        return float(np.dot(a, b))

    def top_k_similar(
        self,
        query_vec: np.ndarray,
        candidate_vecs: np.ndarray,
        k: int = config.TOP_K_TERMS,
        threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> list[tuple[int, float]]:
        """
        Return the indices and scores of the top-k most similar candidates.

        Args:
            query_vec:      Shape (D,).
            candidate_vecs: Shape (N, D).
            k:              Maximum number of results.
            threshold:      Minimum cosine similarity to include.

        Returns:
            Sorted list of (index, score) tuples, descending by score.
        """
        scores: np.ndarray = candidate_vecs @ query_vec
        top_indices = np.argsort(scores)[::-1]
        results: list[tuple[int, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < threshold:
                break
            results.append((int(idx), score))
            if len(results) >= k:
                break
        return results
