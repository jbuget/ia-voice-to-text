# IA Voice to Text

Petit utilitaire en ligne de commande et service API pour transcrire ou traduire un fichier audio/Video en texte avec une implementation locale et open source de Whisper (faster-whisper).

## Prerequis
- Python 3.10 ou plus recent
- Une carte GPU NVIDIA optional selon vos besoins (le script fonctionne aussi sur CPU)

## Installation
1. Creez un environnement virtuel (optionnel mais recommande) :
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Installez les dependances :
   ```bash
   pip install -r requirements.txt
   ```
3. Telechargez un modele Whisper open weight. Par exemple :
   - Utilisez le script fourni :
     ```bash
     chmod +x download_model.sh
     ./download_model.sh Systran/faster-whisper-small models/whisper-small
     ```
     Les fichiers optionnels absents du depot (ex. `tokenizer_config.json`) sont ignores avec un simple avertissement.
   - Telechargement manuel d'un modele 
     (`whisper-small` par defaut) depuis https://huggingface.co/Systran/faster-whisper-small (placez-le dans `./models/whisper-small`).
   - Ou laissez `faster-whisper` recuperer automatiquement le modele en ligne si vous avez un acces reseau.

## Utilisation CLI
```bash
python transcribe.py chemin/vers/fichier_audio.mp3
```

Le script genere un fichier texte cote a cote du fichier source (`fichier_audio.txt`) contenant uniquement la transcription (une ligne par segment reconnu, sans horodatages). Des logs apparaissent au lancement et à la fin pour indiquer le temps de traitement et le nombre de segments/mots/caractères produits.

### Options utiles
- `-o/--output`: chemin du fichier texte de sortie (les dossiers manquants sont créés automatiquement).
- `-m/--model`: chemin vers un dossier de modele local ou nom d'un modele Hugging Face (défaut `./models/whisper-small`).
- `--device`: force `cpu`, `cuda` ou `auto` (auto par defaut).
- `--compute-type`: controle la precision (ex. `int8`, `float16`, `float32`). Par defaut le script choisit `float32` sur CPU pour eviter l'avertissement ctranslate2 et `float16` sur GPU.
- `--language`: force un code langue (ex. `fr`, `en`). Par defaut la detection automatique est active.
- `--translate-to-en`: effectue une traduction vers l'anglais au lieu d'une simple transcription.
- `--word-timestamps`: calcule aussi les horodatages au mot (plus lent) — utile si vous souhaitez modifier le script pour les exploiter.
- `--vad`: active un filtre VAD pour reduire les silences/bruits.

### Exemple pour traduire vers l'anglais
```bash
python transcribe.py chemin/video.mp4 --translate-to-en --device auto
```

### Conseils de performance
- Sur CPU, preferez un modele taille moyenne (`small`, `medium`).
- Sur GPU CUDA, vous pouvez utiliser `--compute-type float16` ou `int8_float16` pour reduire la consommation memoire.

## Utilisation API
1. Assurez-vous que le modele est disponible localement (voir section Installation).
2. Lancez le serveur FastAPI :
   ```bash
   uvicorn app:app --reload
   ```
   Variables d'environnement disponibles :
   - `TRANSCRIBE_MODEL` (defaut `./models/whisper-small`)
   - `TRANSCRIBE_DEVICE` (`auto` | `cpu` | `cuda`)
   - `TRANSCRIBE_COMPUTE_TYPE` (ex. `float16`, `float32`)
3. Envoyez un fichier audio/Video via POST `http://localhost:8000/transcribe` (multipart, champ `file`).
   Vous pouvez transmettre en champs de formulaire les options `language`, `translate_to_en`, `vad`, `word_timestamps`.

Réponse de l'API :
```json
{
  "text": "...",
  "segments": [
    {"start": 0.0, "end": 5.2, "text": "..."}
  ],
  "language": "fr",
  "language_probability": 0.99,
  "word_count": 42,
  "char_count": 210,
  "segment_count": 5,
  "device": "cpu",
  "compute_type": "float32"
}
```

### Integration n8n (exemple rapide)
- Node `HTTP Request` -> POST vers `/transcribe` en multipart avec le fichier audio (`file`).
- Récupérez `{{ $json["text"] }}` pour enchaîner vos traitements (résumé, stockage, etc.).

## Structure du projet
- `transcribe.py` : script principal de transcription/traduction.
- `app.py` : service FastAPI exposant l'API `/transcribe`.
- `models/` : repertory cible pour stocker les modeles telecharges.
- `input/` : placez ici vos fichiers audio/video avant traitement.
- `requirements.txt` : dependances Python.

## Roadmap possible
- Automatisation du telechargement des modeles.
- Interface utilisateur (web/desktop) pour lancer la transcription sans CLI.
- Tests automatises et integration continue.
