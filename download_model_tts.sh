#!/usr/bin/env bash
set -euo pipefail

# Usage: ./download_model_tts.sh [model_id] [target_dir]
# Default model: tts_models/fr/css10/vits
# Default target_dir: ./models (servira de racine cache Coqui TTS via $TTS_HOME)

MODEL_ID="${1:-tts_models/fr/css10/vits}"
TARGET_DIR="${2:-models}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Erreur: interpréteur Python introuvable ('$PYTHON_BIN')." >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"

"$PYTHON_BIN" - "$MODEL_ID" "$TARGET_DIR" <<'PY'
import os
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit("Usage: download_model_tts.py <model_id> [target_dir]")

model_id = sys.argv[1]
target_dir = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

try:
    from TTS.utils.manage import ModelManager
except ModuleNotFoundError as exc:
    raise SystemExit(
        "La bibliothèque 'TTS' est requise. Installez-la avec 'pip install TTS'."
    ) from exc

cache_path = Path(target_dir).expanduser().resolve()

os.environ["TTS_HOME"] = str(cache_path)
cache_path.mkdir(parents=True, exist_ok=True)

print(f"Téléchargement du modèle '{model_id}' dans '{cache_path}'...")
manager = ModelManager()
result = manager.download_model(model_id)
print("Modèle téléchargé avec succès.")
if isinstance(result, (list, tuple)) and result:
    for idx, item in enumerate(result, start=1):
        print(f"- ressource {idx}: {item}")
else:
    print(f"- détail: {result}")
PY
