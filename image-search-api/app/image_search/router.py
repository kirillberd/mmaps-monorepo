from __future__ import annotations

from fastapi import Depends, File, HTTPException, Query, UploadFile
from fastapi.routing import APIRouter
from dependency_injector.wiring import Provide, inject
import anyio

from app.di.container import Container
from app.image_search.schemas import ImageSearchResponse, SearchHit
from app.services.image_search_service import ImageSearchService
from app.services.preprocessing import ImagePreprocessingError

router = APIRouter(prefix="/image-search", tags=["image-search"])


@router.post("/search", response_model=ImageSearchResponse)
@inject
async def search_image(
    file: UploadFile = File(...),
    k: int | None = Query(None, ge=1),
    default_k: int = Depends(Provide[Container.config.search.default_k]),
    max_k: int = Depends(Provide[Container.config.search.max_k]),
    service: ImageSearchService = Depends(Provide[Container.image_search_service]),
):
    image_bytes: bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    effective_k = int(k) if k is not None else int(default_k)
    if effective_k > int(max_k):
        raise HTTPException(status_code=400, detail=f"k must be <= {int(max_k)}")

    try:
        results = await anyio.to_thread.run_sync(service.search, image_bytes, effective_k)
    except ImagePreprocessingError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        # Triton/OpenSearch/other failures should surface as 502
        raise HTTPException(status_code=502, detail=str(e)) from e

    return ImageSearchResponse(
        k=effective_k,
        results=[SearchHit(id=r._id, score=r.score, source=r.source) for r in results],
    )


# Backwards-compatible alias (old endpoint name).
@router.post("/upload", response_model=ImageSearchResponse)
@inject
async def upload_image(
    file: UploadFile = File(...),
    k: int | None = Query(None, ge=1),
    default_k: int = Depends(Provide[Container.config.search.default_k]),
    max_k: int = Depends(Provide[Container.config.search.max_k]),
    service: ImageSearchService = Depends(Provide[Container.image_search_service]),
):
    return await search_image(
        file=file,
        k=k,
        default_k=default_k,
        max_k=max_k,
        service=service,
    )
