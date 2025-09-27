from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from transcribe import create_model
from ..config import Settings

ModelBundle = Tuple[object, str, str]  # WhisperModel, device, compute_type


class ModelNotFoundError(KeyError):
    pass


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._cache: Dict[Tuple[str, str, Optional[str]], ModelBundle] = {}
        self._registry = self._discover_models()
        if not self.settings.default_model_path.is_dir():
            raise RuntimeError(
                f"Le modèle par défaut '{self.settings.default_model_path}' est introuvable. "
                "Téléchargez-le avant de lancer l'API."
            )
        self._default_alias = self._resolve_default_alias()

    @property
    def default_alias(self) -> str:
        return self._default_alias

    @property
    def registry(self) -> Dict[str, str]:
        return dict(self._registry)

    def available_aliases(self) -> List[str]:
        return sorted(self._registry.keys())

    def _key(self, model_path: str) -> Tuple[str, str, Optional[str]]:
        return (
            str(Path(model_path).resolve()),
            self.settings.device_option,
            self.settings.compute_type_option,
        )

    def _discover_models(self) -> Dict[str, str]:
        models: Dict[str, str] = {}
        if self.settings.model_root.exists():
            for entry in self.settings.model_root.iterdir():
                if entry.is_dir():
                    models[entry.name] = str(entry.resolve())
        return models

    def _resolve_default_alias(self) -> str:
        default_path = str(self.settings.default_model_path)
        for alias, path in self._registry.items():
            if Path(path).resolve() == Path(default_path):
                return alias
        alias = Path(default_path).name or "default"
        self._registry.setdefault(alias, default_path)
        return alias

    def list_models(self) -> List[Dict[str, str | bool]]:
        ready = []
        for alias, path in self._registry.items():
            key = self._key(path)
            ready.append({
                "alias": alias,
                "path": path,
                "loaded": key in self._cache,
            })
        return ready

    def load_all(self) -> None:
        errors: Dict[str, str] = {}
        for alias, path in self._registry.items():
            try:
                self._load_model(path)
            except Exception as exc:  # pragma: no cover - defensive
                errors[alias] = str(exc)
        if errors:
            for alias, message in errors.items():
                print(f"[WARN] Échec du chargement du modèle '{alias}': {message}")
        self.get_model(self._default_alias)  # ensure default ready

    async def ensure_loaded_async(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.load_all)

    def _load_model(self, model_path: str) -> ModelBundle:
        key = self._key(model_path)
        with self._lock:
            if key not in self._cache:
                model, device, compute = create_model(
                    model_path,
                    device_option=self.settings.device_option,
                    compute_type_option=self.settings.compute_type_option,
                )
                self._cache[key] = (model, device, compute)
        return self._cache[key]

    def _resolve_selection(self, selection: Optional[str]) -> Tuple[str, str]:
        if selection is None:
            return self._default_alias, self._registry[self._default_alias]
        if selection in self._registry:
            return selection, self._registry[selection]
        candidate = str(Path(selection).resolve())
        for alias, path in self._registry.items():
            if Path(path).resolve() == Path(candidate):
                return alias, path
        raise ModelNotFoundError(selection)

    def get_model(self, selection: Optional[str]) -> Tuple[str, str, ModelBundle]:
        alias, path = self._resolve_selection(selection)
        bundle = self._load_model(path)
        return alias, path, bundle


__all__ = ["ModelManager", "ModelNotFoundError"]
