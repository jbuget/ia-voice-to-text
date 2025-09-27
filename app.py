import asyncio
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from transcribe import create_model, transcribe_audio

MODEL_ROOT = Path(os.getenv("TRANSCRIBE_MODEL_ROOT", "./models")).resolve()
DEFAULT_MODEL_NAME = os.getenv("TRANSCRIBE_DEFAULT_MODEL", "whisper-medium")
DEFAULT_MODEL_PATH = os.getenv(
    "TRANSCRIBE_MODEL", str(MODEL_ROOT / DEFAULT_MODEL_NAME)
)
DEVICE_OPTION = os.getenv("TRANSCRIBE_DEVICE", "auto")
COMPUTE_TYPE_OPTION = os.getenv("TRANSCRIBE_COMPUTE_TYPE") or None

app = FastAPI(title="IA Voice to Text API", version="0.1.0")

ModelKey = Tuple[str, str, Optional[str]]
ModelBundle = Tuple[Any, str, str]

_model_cache: Dict[ModelKey, ModelBundle] = {}
_model_lock = Lock()


def _model_key(model_path: str) -> ModelKey:
    return (str(Path(model_path).resolve()), DEVICE_OPTION, COMPUTE_TYPE_OPTION)


def _discover_local_models() -> Dict[str, str]:
    models: Dict[str, str] = {}
    if MODEL_ROOT.exists():
        for entry in MODEL_ROOT.iterdir():
            if entry.is_dir():
                models[entry.name] = str(entry.resolve())
    return models


MODEL_REGISTRY = _discover_local_models()
DEFAULT_MODEL_RESOLVED = str(Path(DEFAULT_MODEL_PATH).resolve())

if not Path(DEFAULT_MODEL_RESOLVED).is_dir():
    raise RuntimeError(
        f"Le modèle par défaut '{DEFAULT_MODEL_PATH}' est introuvable. "
        "Assurez-vous de l'avoir téléchargé avant de lancer l'API."
    )

DEFAULT_MODEL_ALIAS: Optional[str] = None
for alias, path in list(MODEL_REGISTRY.items()):
    if path == DEFAULT_MODEL_RESOLVED:
        DEFAULT_MODEL_ALIAS = alias
        break

if DEFAULT_MODEL_ALIAS is None:
    DEFAULT_MODEL_ALIAS = Path(DEFAULT_MODEL_RESOLVED).name or "default"
    MODEL_REGISTRY.setdefault(DEFAULT_MODEL_ALIAS, DEFAULT_MODEL_RESOLVED)

AVAILABLE_MODEL_NAMES = sorted(MODEL_REGISTRY.keys())


def _resolve_model_selection(selection: Optional[str]) -> Tuple[str, str]:
    if selection is None:
        alias = DEFAULT_MODEL_ALIAS
    else:
        if selection in MODEL_REGISTRY:
            alias = selection
        else:
            candidate = str(Path(selection).resolve())
            alias = next(
                (
                    name
                    for name, path in MODEL_REGISTRY.items()
                    if path == candidate
                ),
                None,
            )
            if alias is None:
                raise KeyError(selection)
    if alias is None:
        raise KeyError("default")
    return alias, MODEL_REGISTRY[alias]


def _ensure_model_loaded(model_path: str) -> ModelBundle:
    key = _model_key(model_path)
    with _model_lock:
        if key not in _model_cache:
            _model_cache[key] = create_model(
                model_path,
                device_option=DEVICE_OPTION,
                compute_type_option=COMPUTE_TYPE_OPTION,
            )
        return _model_cache[key]


def _get_cached_model(model_path: str) -> ModelBundle:
    key = _model_key(model_path)
    with _model_lock:
        bundle = _model_cache.get(key)
    if bundle is None:
        raise KeyError(model_path)
    return bundle


@app.on_event("startup")
def load_models() -> None:
    errors: Dict[str, str] = {}
    for alias, path in MODEL_REGISTRY.items():
        try:
            _ensure_model_loaded(path)
        except Exception as exc:  # pragma: no cover - defensive
            errors[alias] = str(exc)

    try:
        _get_cached_model(MODEL_REGISTRY[DEFAULT_MODEL_ALIAS])
    except KeyError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Impossible de charger le modèle par défaut '{DEFAULT_MODEL_ALIAS}'."
        ) from exc

    if errors:
        for alias, message in errors.items():
            print(
                f"[WARN] Échec du chargement du modèle '{alias}': {message}",
                flush=True,
            )


@app.get("/health")
def health_check() -> dict:
    default_path = MODEL_REGISTRY[DEFAULT_MODEL_ALIAS]
    key = _model_key(default_path)
    bundle = _model_cache.get(key)
    status = "ok" if bundle else "loading"
    loaded = [
        {
            "alias": alias,
            "path": path,
            "loaded": _model_key(path) in _model_cache,
        }
        for alias, path in MODEL_REGISTRY.items()
    ]
    return {
        "status": status,
        "default_model": {
            "alias": DEFAULT_MODEL_ALIAS,
            "path": default_path,
            "device": bundle[1] if bundle else None,
            "compute_type": bundle[2] if bundle else None,
        },
        "loaded_models": loaded,
    }


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    vad: bool = Form(False),
    word_timestamps: bool = Form(False),
) -> dict:
    try:
        model_alias, model_path = _resolve_model_selection(model)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=
            "Modèle inconnu. Modèles disponibles: "
            + ", ".join(AVAILABLE_MODEL_NAMES),
        )

    try:
        model_instance, model_device, model_compute_type = _get_cached_model(
            model_path
        )
    except KeyError:
        raise HTTPException(
            status_code=503,
            detail=f"Le modèle '{model_alias}' n'est pas prêt. Consultez /health.",
        )

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
