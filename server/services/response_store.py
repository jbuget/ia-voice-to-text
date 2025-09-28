from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List, Optional


class ResponseStore:
    def __init__(self, max_history: int = 20) -> None:
        self._lock = Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._history: List[Dict[str, Any]] = []
        self._max_history = max_history

    def add(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._latest = payload
            self._history.append(payload)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._latest is None:
                return None
            return dict(self._latest)

    def history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._history]


__all__ = ["ResponseStore"]
