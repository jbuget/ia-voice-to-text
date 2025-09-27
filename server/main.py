from fastapi import FastAPI

from .config import Settings
from .services.model_manager import ModelManager
from .routes import api, pages


def create_app() -> FastAPI:
    settings = Settings()
    model_manager = ModelManager(settings)

    app = FastAPI(title="IA Voice to Text API", version="0.2.0")
    app.state.settings = settings
    app.state.model_manager = model_manager

    @app.on_event("startup")
    def on_startup() -> None:
        model_manager.load_all()

    app.include_router(pages.router)
    app.include_router(api.router)

    return app


app = create_app()


__all__ = ["app", "create_app"]
