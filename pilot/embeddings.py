"""Local MiniLM embeddings. Same model and normalization as notebook cell 6."""

from __future__ import annotations

import numpy as np

_model = None


def embed(texts: list[str]) -> np.ndarray:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return _model.encode(
        texts,
        batch_size=256,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)


def max_pairwise(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return float("nan")
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -1)
    return float(similarity.max())


def nearest_neighbor(vectors: np.ndarray) -> np.ndarray:
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -1)
    return similarity.max(axis=1)
