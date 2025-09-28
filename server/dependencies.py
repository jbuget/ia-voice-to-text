from fastapi import Request

from .config import Settings
from .services.model_manager import ModelManager
from .services.response_store import ResponseStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[attr-defined]


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager  # type: ignore[attr-defined]


def get_response_store(request: Request) -> ResponseStore:
    return request.app.state.response_store  # type: ignore[attr-defined]


__all__ = ["get_settings", "get_model_manager", "get_response_store"]
