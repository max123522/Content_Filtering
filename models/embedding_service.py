"""
embedding_service.py — Semantic embedding for the IAI DLP system.

DEV  mode: model runs locally via SentenceTransformer (no external calls).
PROD mode: delegates to the IAI on-prem embedding endpoint defined by
           PROD_LLM_BASE_URL_EMBED (OpenAI-compatible /v1/embeddings API).

Thread safety
-------------
* Singleton instantiation is protected by a class-level Lock (double-checked
  locking pattern) so that multiple threads starting simultaneously cannot
  initialise the model twice.
* In DEV mode, local CPU inference is serialised by _encode_lock — PyTorch CPU
  operations are technically thread-safe but serialisation prevents resource
  contention on the inference worker threads.
* In PROD mode, each HTTP call to the remote endpoint is independent; no lock
  is needed.

Embeddings are L2-normalised before storage / comparison.
"""

from __future__ import annotations

import threading

import numpy as np

import config


class EmbeddingService:
    """
    Singleton embedding service.

    In DEV the model is loaded locally; in PROD every encode call is
    forwarded to the IAI on-prem OpenAI-compatible endpoint so that no
    model weights need to be shipped to the production server.
    """

    _instance: EmbeddingService | None = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> EmbeddingService:
        # Fast path — no lock acquisition when already initialised.
        if cls._instance is None:
            with cls._singleton_lock:
                # Second check inside the lock (double-checked locking).
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialised = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return

        with self._singleton_lock:
            if self._initialised:
                return

            self._prod_mode: bool = (
                config.ENV == "PROD" and bool(config.EMBED_BASE_URL)
            )

            if self._prod_mode:
                # Lazy import — openai is only required in PROD.
                from openai import OpenAI  # noqa: PLC0415

                self._openai_client = OpenAI(
                    base_url=config.EMBED_BASE_URL,
                    api_key=config.EMBED_API_KEY,
                )
                self._embed_model = config.EMBED_MODEL
                print(
                    f"[EmbeddingService] PROD mode — remote endpoint: "
                    f"{config.EMBED_BASE_URL}  model: {self._embed_model}"
                )
            else:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415

                print(
                    f"[EmbeddingService] DEV mode — loading local model: "
                    f"{config.EMBEDDING_MODEL_NAME} …"
                )
                self._model = SentenceTransformer(
                    config.EMBEDDING_MODEL_NAME,
                    trust_remote_code=True,
                    device="cpu",
                )
                model_lower = config.EMBEDDING_MODEL_NAME.lower()
                self._query_prefix = "query: " if "e5" in model_lower else ""
                self._passage_prefix = "passage: " if "e5" in model_lower else ""
                # Serialise local CPU inference — prevents contention when
                # multiple threads call encode concurrently.
                self._encode_lock = threading.Lock()

            self._initialised = True
            print("[EmbeddingService] Ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_remote(self, texts: list[str]) -> np.ndarray:
        """
        Call the IAI on-prem OpenAI-compatible embeddings endpoint.

        Args:
            texts: One or more strings to embed (UTF-8, Hebrew/English).

        Returns:
            np.ndarray of shape (N, D), dtype float32, L2-normalised.
        """
        response = self._openai_client.embeddings.create(
            model=self._embed_model,
            input=texts,
            encoding_format="float",
        )
        vecs = np.array(
            [item.embedding for item in response.data], dtype=np.float32
        )
        # L2-normalise so cosine similarity == dot product (same as local path).
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vecs / norms).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str, is_query: bool = True) -> np.ndarray:
        """
        Return a normalised 1-D float32 embedding for *text*.

        In PROD the call is forwarded to the on-prem embedding endpoint.
        In DEV the local SentenceTransformer is used (serialised by lock).

        Args:
            text:     Input string (Hebrew or English, any length).
            is_query: True when encoding a search query; False for a passage.

        Returns:
            np.ndarray of shape (D,), dtype float32, L2-norm ≈ 1.0.
        """
        if self._prod_mode:
            return self._encode_remote([text])[0]

        prefix = self._query_prefix if is_query else self._passage_prefix
        with self._encode_lock:
            vec: np.ndarray = self._model.encode(
                prefix + text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return vec.astype(np.float32)

    def encode_batch(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """
        Batch-encode a list of strings in a single call.

        In PROD the entire batch is sent in one HTTP request to the
        on-prem endpoint.  In DEV the local model processes the batch
        (serialised by lock so only one batch runs at a time on CPU).

        Returns:
            np.ndarray of shape (N, D), dtype float32.
        """
        if self._prod_mode:
            return self._encode_remote(texts)

        prefix = self._query_prefix if is_query else self._passage_prefix
        prefixed = [prefix + t for t in texts]
        with self._encode_lock:
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
