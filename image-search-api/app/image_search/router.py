from fastapi.routing import APIRouter
from fastapi import UploadFile, File

router = APIRouter(prefix="/image-search", tags=["image-search"])


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_bytes: bytes = await file.read()

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(image_bytes),
    }
