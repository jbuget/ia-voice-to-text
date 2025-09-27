import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from transcribe import transcribe_audio
from ..dependencies import get_model_manager
from ..services.model_manager import ModelManager, ModelNotFoundError

router = APIRouter()


@router.get("/health")
def health(model_manager: ModelManager = Depends(get_model_manager)) -> dict:
    default_alias = model_manager.default_alias
    registry = model_manager.registry
    default_path = registry[default_alias]
    list_models = model_manager.list_models()
    default_bundle = next(
        (item for item in list_models if item["alias"] == default_alias),
        None,
    )
    device = None
    compute = None
    if default_bundle and default_bundle["loaded"]:
        _, _, bundle = model_manager.get_model(default_alias)
        device = bundle[1]
        compute = bundle[2]
    return {
        "status": "ok" if default_bundle and default_bundle["loaded"] else "loading",
        "default_model": {
            "alias": default_alias,
            "path": default_path,
            "device": device,
            "compute_type": compute,
        },
        "loaded_models": list_models,
    }


@router.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    vad: bool = Form(False),
    word_timestamps: bool = Form(False),
    model_manager: ModelManager = Depends(get_model_manager),
) -> dict:
    try:
        model_alias, model_path, bundle = model_manager.get_model(model)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Modèle inconnu. Modèles disponibles: "
                + ", ".join(model_manager.available_aliases())
            ),
        )

    model_instance, model_device, model_compute_type = bundle

    filename = file.filename or "audio"
    suffix = Path(filename).suffix or ".tmp"

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        result = await asyncio.to_thread(
            transcribe_audio,
            model=model_instance,
            audio_path=tmp_path,
            language=language or None,
            vad=vad,
            word_timestamps=word_timestamps,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "text": result.text,
        "segments": result.segments,
        "language": result.info.get("language"),
        "language_probability": result.info.get("language_probability"),
        "word_count": result.word_count,
        "char_count": result.char_count,
        "segment_count": result.segment_count,
        "model": model_alias,
        "model_path": model_path,
        "device": model_device,
        "compute_type": model_compute_type,
    }
