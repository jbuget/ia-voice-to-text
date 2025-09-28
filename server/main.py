from fastapi import FastAPI

from .config import Settings
from .services.model_manager import ModelManager
from .services.response_store import ResponseStore
from .routes import api, pages
from contextlib import asynccontextmanager


def create_app() -> FastAPI:
    settings = Settings()
    model_manager = ModelManager(settings)
    response_store = ResponseStore()

    app = FastAPI(title="IA Voice to Text API", version="0.2.0")
    app.state.settings = settings
    app.state.model_manager = model_manager
    app.state.response_store = response_store

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model_manager.load_all()
        yield

    app.router.lifespan_context = lifespan

    app.include_router(pages.router)
    app.include_router(api.router)

    return app


app = create_app()


__all__ = ["app", "create_app"]
