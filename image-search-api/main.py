from fastapi import FastAPI
import uvicorn
from app.config import get_settings


app = FastAPI()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
    )
