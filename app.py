import asyncio
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from transcribe import create_model, transcribe_audio

MODEL_PATH = os.getenv("TRANSCRIBE_MODEL", "./models/whisper-small")
DEVICE_OPTION = os.getenv("TRANSCRIBE_DEVICE", "auto")
COMPUTE_TYPE_OPTION = os.getenv("TRANSCRIBE_COMPUTE_TYPE") or None

app = FastAPI(title="IA Voice to Text API", version="0.1.0")

_model = None
_model_device = None
_model_compute_type = None


@app.on_event("startup")
def load_model() -> None:
    global _model, _model_device, _model_compute_type
    _model, _model_device, _model_compute_type = create_model(
        MODEL_PATH,
        device_option=DEVICE_OPTION,
        compute_type_option=COMPUTE_TYPE_OPTION,
    )


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok" if _model is not None else "loading",
        "model": MODEL_PATH,
        "device": _model_device,
        "compute_type": _model_compute_type,
    }


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    translate_to_en: bool = Form(False),
    vad: bool = Form(False),
    word_timestamps: bool = Form(False),
) -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    filename = file.filename or "audio"
    suffix = Path(filename).suffix or ".tmp"

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        result = await asyncio.to_thread(
            transcribe_audio,
            model=_model,
            audio_path=tmp_path,
            language=language or None,
            translate_to_en=translate_to_en,
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
        "device": _model_device,
        "compute_type": _model_compute_type,
    }
