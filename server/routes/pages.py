from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates

from ..dependencies import get_model_manager, get_settings
from ..services.model_manager import ModelManager
from ..config import Settings

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/upload")
def upload_page(
    request: Request,
    settings: Settings = Depends(get_settings),
    model_manager: ModelManager = Depends(get_model_manager),
):
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "forward_url": settings.forward_url,
            "model_aliases": model_manager.available_aliases(),
            "default_alias": model_manager.default_alias,
        },
    )


@router.get("/recording")
def recording_page(
    request: Request,
    model_manager: ModelManager = Depends(get_model_manager),
):
    return templates.TemplateResponse(
        "recording.html",
        {
            "request": request,
            "model_aliases": model_manager.available_aliases(),
            "default_alias": model_manager.default_alias,
        },
    )
