"""Utilities for turning text into spoken audio using Coqui TTS."""
from __future__ import annotations

import functools
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional

from TTS.api import TTS


def _ensure_default_cache() -> None:
    default_home = Path(os.getenv("TTS_HOME", "models")).expanduser().resolve()
    os.environ.setdefault("TTS_HOME", str(default_home))
    default_home.mkdir(parents=True, exist_ok=True)

    tts_dir = default_home / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    nested = tts_dir / "tts"
    if nested.exists() and nested.is_dir():
        for child in nested.iterdir():
            target = tts_dir / child.name
            if target.exists():
                continue
            shutil.move(str(child), target)
        try:
            nested.rmdir()
        except OSError:
            pass


_ensure_default_cache()


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


def _available_languages(tts: TTS) -> List[str]:
    languages = getattr(tts, "languages", None)
    if not languages:
        return []
    if isinstance(languages, dict):
        languages = list(languages.keys())
    return [str(lang).lower() for lang in languages]


def _available_speakers(tts: TTS) -> List[str]:
    speakers = getattr(tts, "speakers", None)
    if not speakers:
        return []
    return [str(speaker) for speaker in speakers]


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

    selected_language = language.strip().lower() if language else None
    kwargs: Dict[str, Optional[str]] = {}

    supported_languages = _available_languages(tts)
    if selected_language:
        if supported_languages:
            if selected_language not in supported_languages:
                raise TextToSpeechError(
                    "Langue '"
                    + selected_language
                    + "' non supportée par le modèle. Langues disponibles: "
                    + ", ".join(sorted(supported_languages))
                )
            kwargs["language"] = selected_language
        # if model is mono-language, silently ignore the language hint

    if speaker_id:
        supported_speakers = _available_speakers(tts)
        if not supported_speakers:
            raise TextToSpeechError(
                "Le modèle sélectionné ne propose pas de voix alternatives."
            )
        if speaker_id not in supported_speakers:
            raise TextToSpeechError(
                "Voix '"
                + speaker_id
                + "' inconnue. Voix disponibles: "
                + ", ".join(sorted(supported_speakers))
            )
        kwargs["speaker"] = speaker_id

    try:
        with NamedTemporaryFile(suffix=".wav") as tmp:
            tts.tts_to_file(
                text=text,
                file_path=tmp.name,
                **kwargs,
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
