from fastapi import FastAPI
import uvicorn
from app.config import get_settings
from app.image_search import router as image_search_router

app = FastAPI()
app.include_router(image_search_router)

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
    )
