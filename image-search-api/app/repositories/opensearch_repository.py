from __future__ import annotations

from typing import Any

import numpy as np

try:
    from opensearchpy import OpenSearch
except Exception:  # noqa: BLE001
    OpenSearch = None  # type: ignore


class OpenSearchClientNotInstalledError(RuntimeError):
    pass


class OpenSearchQueryError(RuntimeError):
    pass


class OpenSearchRepository:
    """Thin wrapper around OpenSearch kNN search."""

    def __init__(
        self,
        host: str,
        port: int,
        index: str,
        username: str | None = None,
        password: str | None = None,
        use_ssl: bool = False,
        verify_certs: bool = False,
        http_compress: bool = True,
        timeout_s: int = 600,
        request_timeout_s: int = 60,
        ef_search: int = 128,
        source_includes: list[str] | None = None,
    ) -> None:
        if OpenSearch is None:
            raise OpenSearchClientNotInstalledError(
                "opensearch-py is not installed. Add dependency: opensearch-py",
            )

        self.host = host
        self.port = int(port)
        self.index = index
        self.ef_search = int(ef_search)
        self.request_timeout_s = int(request_timeout_s)
        self.source_includes = source_includes

        http_auth = None
        if username is not None or password is not None:
            http_auth = (username or "", password or "")

        self._client = OpenSearch(
            hosts=[{"host": self.host, "port": self.port}],
            http_auth=http_auth,
            use_ssl=bool(use_ssl),
            verify_certs=bool(verify_certs),
            ssl_show_warn=False,
            http_compress=bool(http_compress),
            timeout=int(timeout_s),
        )

    def search_knn(self, vector: np.ndarray, k: int) -> list[dict[str, Any]]:
        """Returns OpenSearch hits (id + _source + score)."""

        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1:
            raise ValueError(f"Expected 1D vector, got shape {vector.shape}")

        body: dict[str, Any] = {
            "size": int(k),
            "query": {
                "knn": {
                    "vec": {
                        "vector": vector.astype(np.float32).tolist(),
                        "k": int(k),
                        "method_parameters": {"ef_search": int(self.ef_search)},
                    }
                }
            },
        }

        if self.source_includes:
            body["_source"] = {"includes": list(self.source_includes)}

        try:
            res = self._client.search(
                index=self.index,
                body=body,
                request_timeout=self.request_timeout_s,
            )
        except Exception as e:  # noqa: BLE001
            raise OpenSearchQueryError(f"OpenSearch query failed: {e}") from e

        hits = res.get("hits", {}).get("hits", [])
        out: list[dict[str, Any]] = []
        for h in hits:
            out.append(
                {
                    "_id": h.get("_id"),
                    "score": h.get("_score"),
                    "source": h.get("_source", {}),
                }
            )
        return out
