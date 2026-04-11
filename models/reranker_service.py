"""
reranker_service.py — Local cross-encoder reranker using BGE-Reranker-Base.

The reranker cross-checks the LLM's preliminary decision by scoring the
relevance of a text segment against each matched term's context description.
A high reranker score confirms the LLM decision; a low score may indicate
a false alarm.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

import config


class RerankerService:
    """Singleton wrapper around the BGE-Reranker-Base cross-encoder model."""

    _instance: RerankerService | None = None

    def __new__(cls) -> RerankerService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        print(
            f"[RerankerService] Loading model: {config.RERANKER_MODEL_NAME} …"
        )
        self._model = CrossEncoder(
            config.RERANKER_MODEL_NAME,
            max_length=512,
        )
        self._initialised = True
        print("[RerankerService] Model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score(self, query: str, passage: str) -> float:
        """
        Return a relevance score in [0, 1] between *query* and *passage*.

        A score above 0.5 indicates the passage is semantically related to
        the query context.

        Args:
            query:   The context description of a forbidden term (English).
            passage: The document segment being evaluated.

        Returns:
            Float relevance score.
        """
        raw: float = self._model.predict([[query, passage]])
        # Sigmoid normalisation for raw logit outputs
        score = float(1.0 / (1.0 + np.exp(-raw)))
        return score

    def score_batch(
        self, queries: list[str], passages: list[str]
    ) -> list[float]:
        """
        Score multiple (query, passage) pairs in one forward pass.

        Args:
            queries:  List of context descriptions.
            passages: List of document segments (same length as queries).

        Returns:
            List of float scores.
        """
        pairs = list(zip(queries, passages))
        raw_scores: np.ndarray = self._model.predict(pairs)
        return [float(1.0 / (1.0 + np.exp(-s))) for s in raw_scores]

    def rerank_candidates(
        self,
        segment: str,
        candidates: list[tuple[str, str]],  # [(term, context_description), ...]
        threshold: float = 0.4,
    ) -> list[tuple[str, float]]:
        """
        Rerank candidate forbidden terms against a document segment.

        Returns:
            List of (term, reranker_score) sorted descending, above threshold.
        """
        if not candidates:
            return []
        queries = [ctx for _, ctx in candidates]
        passages = [segment] * len(queries)
        scores = self.score_batch(queries, passages)
        results = [
            (term, score)
            for (term, _), score in zip(candidates, scores)
            if score >= threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
