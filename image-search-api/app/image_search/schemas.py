from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    id: str | None = Field(default=None, description="OpenSearch document _id")
    score: float | None = None
    source: dict[str, Any] = Field(default_factory=dict)


class ImageSearchResponse(BaseModel):
    k: int
    results: list[SearchHit]