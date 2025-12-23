from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.clients.triton_embedder import TritonEmbedder
from app.repositories.opensearch_repository import OpenSearchRepository
from app.services.preprocessing import ImagePreprocessor


@dataclass(slots=True)
class ImageSearchResult:
    _id: str | None
    score: float | None
    source: dict[str, Any]


class ImageSearchService:
    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        embedder: TritonEmbedder,
        repository: OpenSearchRepository,
    ) -> None:
        self._preprocessor = preprocessor
        self._embedder = embedder
        self._repository = repository

    def search(self, image_bytes: bytes, k: int) -> list[ImageSearchResult]:
        # 1) preprocess -> [1,3,224,224]
        images = self._preprocessor.preprocess(image_bytes)

        # 2) triton -> [1,1280]
        embeddings = self._embedder.embed(images)
        if embeddings.ndim == 2:
            if embeddings.shape[0] != 1:
                raise ValueError(
                    f"Expected embeddings shape [1, D], got {embeddings.shape}"
                )
            q: np.ndarray = embeddings[0]
        elif embeddings.ndim == 1:
            q = embeddings
        else:
            raise ValueError(f"Unexpected embeddings shape: {embeddings.shape}")

        # 3) OpenSearch KNN
        hits = self._repository.search_knn(q, k=k)
        return [
            ImageSearchResult(
                _id=str(h.get("_id")) if h.get("_id") is not None else None,
                score=float(h.get("score")) if h.get("score") is not None else None,
                source=dict(h.get("source") or {}),
            )
            for h in hits
        ]
