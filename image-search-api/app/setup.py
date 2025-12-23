from fastapi import FastAPI

from app.config import Settings
from app.di.container import Container
import app.image_search.router as image_search_router_module


def setup(settings: Settings) -> FastAPI:

    container = Container()

    container.config.from_pydantic(settings)

    container.wire(modules=[image_search_router_module])

    app = FastAPI()
    app.extra["container"] = container
    app.include_router(image_search_router_module.router)
    return app
