import uvicorn
from app import setup
from app.config import Settings
from pathlib import Path

settings = Settings.from_yaml(Path(__file__).resolve().parent / "config" / "config.yml")

app = setup.setup(settings)
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
    )
