from fastapi import FastAPI
from app.di.container import Container
from app.config import Settings
from app.image_search import router as image_search_router


def setup(settings: Settings) -> FastAPI:

    container = Container()

    container.config.from_pydantic(settings)

    app = FastAPI()
    app.extra["container"] = container
    app.include_router(image_search_router)
    return app
