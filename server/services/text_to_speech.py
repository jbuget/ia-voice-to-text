"""Utilities for turning text into spoken audio using Coqui TTS."""
from __future__ import annotations

import functools
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

from TTS.api import TTS


class TextToSpeechError(RuntimeError):
    """Raised when text-to-speech synthesis fails."""


DEFAULT_LANGUAGE = "fr"
MODEL_BY_LANGUAGE: Dict[str, str] = {
    "fr": "tts_models/fr/css10/vits",
    "en": "tts_models/en/vctk/vits",
}
DEFAULT_MODEL = MODEL_BY_LANGUAGE[DEFAULT_LANGUAGE]


@functools.lru_cache(maxsize=4)
def _load_model(model_name: str) -> TTS:
    try:
        return TTS(model_name=model_name, progress_bar=False, gpu=False)
    except Exception as exc:  # pragma: no cover - defensive
        raise TextToSpeechError(
            f"Impossible de charger le modèle Coqui TTS '{model_name}': {exc}"
        ) from exc


def _resolve_model(language: Optional[str], model_name: Optional[str]) -> str:
    if model_name:
        return model_name

    if language:
        normalized = language.strip().lower()
        if normalized in MODEL_BY_LANGUAGE:
            return MODEL_BY_LANGUAGE[normalized]

    return DEFAULT_MODEL


def synthesize_to_wav(
    text: str,
    *,
    language: Optional[str] = None,
    model_name: Optional[str] = None,
    speaker_id: Optional[str] = None,
) -> bytes:
    """Generate a WAV payload from the provided text using Coqui TTS."""
    if not text or not text.strip():
        raise TextToSpeechError("Le texte à synthétiser est vide.")

    resolved_model = _resolve_model(language, model_name)
    tts = _load_model(resolved_model)

    try:
        with NamedTemporaryFile(suffix=".wav") as tmp:
            tts.tts_to_file(
                text=text,
                file_path=tmp.name,
                speaker=speaker_id,
                language=language,
            )
            tmp.seek(0)
            return tmp.read()
    except RuntimeError as exc:  # pragma: no cover - depends on TTS backend
        raise TextToSpeechError(f"Erreur lors de la synthèse vocale: {exc}") from exc


__all__ = [
    "TextToSpeechError",
    "synthesize_to_wav",
    "DEFAULT_MODEL",
    "MODEL_BY_LANGUAGE",
]
