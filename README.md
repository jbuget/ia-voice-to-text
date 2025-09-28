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
     chmod +x download_model_stt.sh
     ./download_model_stt.sh Systran/faster-whisper-medium models/stt/whisper-medium
     ```
     Les fichiers optionnels absents du depot (ex. `tokenizer_config.json`) sont ignores avec un simple avertissement.
   - Telechargement manuel d'un modele 
     (`whisper-medium` par defaut) depuis https://huggingface.co/Systran/faster-whisper-medium (placez-le dans `./models/stt/whisper-medium`).
   - Ou laissez `faster-whisper` recuperer automatiquement le modele en ligne si vous avez un acces reseau.

## Utilisation CLI
```bash
python transcribe.py chemin/vers/fichier_audio.mp3
```

Le script genere un fichier texte cote a cote du fichier source (`fichier_audio.txt`) contenant uniquement la transcription (une ligne par segment reconnu, sans horodatages). Des logs apparaissent au lancement et à la fin pour indiquer le temps de traitement et le nombre de segments/mots/caractères produits.

### Options utiles
- `-o/--output`: chemin du fichier texte de sortie (les dossiers manquants sont créés automatiquement).
- `-m/--model`: chemin vers un dossier de modele local ou nom d'un modele Hugging Face (défaut `./models/stt/whisper-medium`).
- `--device`: force `cpu`, `cuda` ou `auto` (auto par defaut).
- `--compute-type`: controle la precision (ex. `int8`, `float16`, `float32`). Par defaut le script choisit `float32` sur CPU pour eviter l'avertissement ctranslate2 et `float16` sur GPU.
- `--language`: force un code langue (ex. `fr`, `en`). Par defaut la detection automatique est active.
- `--word-timestamps`: calcule aussi les horodatages au mot (plus lent) — utile si vous souhaitez modifier le script pour les exploiter.
- `--vad`: active un filtre VAD pour reduire les silences/bruits.

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
   - `TRANSCRIBE_MODEL` (defaut `./models/stt/whisper-medium`)
   - `TRANSCRIBE_DEVICE` (`auto` | `cpu` | `cuda`)
   - `TRANSCRIBE_COMPUTE_TYPE` (ex. `float16`, `float32`)
3. Envoyez un fichier audio/Video via POST `http://localhost:8000/transcribe` (multipart, champ `file`).
   - Les modèles exploitables sont ceux **déjà présents dans le dossier `models/stt/` au démarrage** (par exemple `whisper-medium`, `whisper-small`).
   - Le champ `model` accepte le nom du dossier (`whisper-medium`) ou son chemin (`models/stt/whisper-medium`).
   - Vous pouvez transmettre `language`, `vad`, `word_timestamps`.
   - Après avoir ajouté un nouveau modèle sur disque, redémarrez le serveur pour qu'il soit pris en compte.
4. Convertissez un texte en audio via POST `http://localhost:8000/text-to-speech` en JSON :
   ```json
   {
     "text": "Bonjour !",
     "language": "fr",
     "model": "tts_models/fr/css10/vits"
   }
   ```
   - Réponse : un flux binaire WAV (`audio/wav`) téléchargeable.
   - Dépendance : la route s'appuie sur [Coqui TTS](https://github.com/coqui-ai/TTS). Le modèle est téléchargé une première fois (connexion requise) puis la synthèse fonctionne hors ligne.
   - Les champs `language` et `speaker` sont optionnels. Ils ne sont pris en compte que si le modèle choisi est multilingue et/ou multi-voix.
   - Vous pouvez pré-télécharger un modèle via `./download_model_tts.sh tts_models/fr/css10/vits [dossier_cache]`. Sans second argument, le cache racine est `./models` (les modèles sont stockés dans `./models/tts/…`).
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
  "model": "whisper-medium",
  "model_path": "/chemin/absolu/vers/models/stt/whisper-medium",
  "device": "cpu",
  "compute_type": "float32"
}
```

### Integration n8n (exemple rapide)
- Node `HTTP Request` -> POST vers `/transcribe` en multipart avec le fichier audio (`file`).
- Récupérez `{{ $json["text"] }}` pour enchaîner vos traitements (résumé, stockage, etc.).
- Vérifiez `GET /health` pour connaître la liste des modèles chargés (`loaded_models`).
- Option pratique : `GET /upload` sert une page HTML pour envoyer un fichier audio existant vers votre webhook n8n (l'URL cible peut être saisie ou pré-remplie via `TRANSCRIBE_FORWARD_URL`).
- Interface locale : `GET /recording` ouvre une page de capture audio (start/stop/envoi) qui poste directement vers `/transcribe`. La capture repose sur `MediaRecorder` (Chrome, Edge, Firefox, Android). Safari iOS ne gère pas encore cette API : privilégiez un envoi via l'app Dictaphone ou un raccourci iOS.
- Webhook retour : n8n peut notifier l'application via `POST /webhook/response` (JSON libre). Le dernier message reçu est visible sur `/recording` ou via `GET /responses/latest`.

## Structure du projet
- `transcribe.py` : script principal de transcription sur la ligne de commande.
- `app.py` : point d'entrée FastAPI minimal (`uvicorn app:app`) qui délègue à `server/`.
- `server/` : code serveur organisé (config, routes API, vues HTML, gestion des modèles).
- `templates/` : gabarits Jinja2 pour les pages `/upload` et `/recording`.
- `models/` : répertoire pour stocker les modèles téléchargés.
- `input/` : déposez vos fichiers audio/vidéo avant traitement.
- `requirements.txt` : dépendances Python.

## Roadmap possible
- Automatisation du telechargement des modeles.
- Interface utilisateur (web/desktop) pour lancer la transcription sans CLI.
- Tests automatises et integration continue.
