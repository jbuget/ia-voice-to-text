#!/usr/bin/env bash
set -euo pipefail

# Usage: ./download_model.sh [huggingface_repo] [target_dir]
# Example: ./download_model.sh Systran/faster-whisper-large-v3 models/whisper-large-v3

REPO="${1:-Systran/faster-whisper-large-v3}"
TARGET_DIR="${2:-models/whisper-large-v3}"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"
REQUIRED_FILES=(
  config.json
  tokenizer.json
  model.bin
)
VOCAB_CANDIDATES=(
  vocabulary.json
  vocabulary.txt
)
OPTIONAL_FILES=(
  special_tokens_map.json
  tokenizer_config.json
  preprocessor_config.json
)

download() {
  local url="$1"
  local output="$2"

  if command -v curl >/dev/null 2>&1; then
    if curl -L --fail --progress-bar "$url" -o "$output"; then
      return 0
    fi
  elif command -v wget >/dev/null 2>&1; then
    if wget -O "$output" "$url"; then
      return 0
    fi
  else
    echo "Erreur: ni curl ni wget n'est disponible." >&2
    exit 1
  fi

  return 1
}

mkdir -p "$TARGET_DIR"

echo "Téléchargement du modèle \"$REPO\" dans \"$TARGET_DIR\"..."
missing_required=()
for file in "${REQUIRED_FILES[@]}"; do
  echo "- $file"
  if ! download "$BASE_URL/$file" "$TARGET_DIR/$file"; then
    missing_required+=("$file")
  fi
done

vocab_found=false
for file in "${VOCAB_CANDIDATES[@]}"; do
  echo "- $file (vocabulaire)"
  if download "$BASE_URL/$file" "$TARGET_DIR/$file"; then
    vocab_found=true
    break
  else
    rm -f "$TARGET_DIR/$file"
  fi
done

if ((${#missing_required[@]})) || [ "$vocab_found" = false ]; then
  if [ "$vocab_found" = false ]; then
    missing_required+=("vocabulary.json|vocabulary.txt")
  fi
  echo "Erreur: fichiers indispensables introuvables: ${missing_required[*]}" >&2
  exit 1
fi

for file in "${OPTIONAL_FILES[@]}"; do
  echo "- $file (optionnel)"
  if ! download "$BASE_URL/$file" "$TARGET_DIR/$file"; then
    echo "Avertissement: impossible de télécharger $file (peut être absent du dépôt)." >&2
  fi
done

echo "Modèle téléchargé avec succès."
