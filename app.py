import asyncio
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

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


@app.get("/recording", response_class=HTMLResponse)
def recording_page() -> HTMLResponse:
    models_options = "".join(
        f"<option value='{alias}' {'selected' if alias == DEFAULT_MODEL_ALIAS else ''}>{alias}</option>"
        for alias in AVAILABLE_MODEL_NAMES
    )

    template = """<!DOCTYPE html>
<html lang='fr'>
<head>
  <meta charset='utf-8'>
  <title>Enregistrement vocal</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f7f7f7; color: #1f1f1f; }
    main { display: flex; min-height: 100vh; }
    section { flex: 1; padding: 2rem; display: flex; flex-direction: column; gap: 1.5rem; }
    .left { background: #ffffff; border-right: 1px solid #dee2e6; }
    .right { background: #0b7285; color: #fff; overflow-y: auto; }
    h1 { margin: 0; font-size: 1.8rem; }
    p { margin: 0; line-height: 1.5; }
    .controls { display: flex; gap: 1rem; flex-wrap: wrap; }
    button { padding: 0.75rem 1.5rem; border: 0; border-radius: 8px; background: #0b7285; color: #fff; font-size: 1rem; cursor: pointer; transition: background 0.2s ease; }
    button.secondary { background: #495057; }
    button:hover:not(:disabled) { background: #095d6b; }
    button.secondary:hover:not(:disabled) { background: #343a40; }
    button:disabled { background: #adb5bd; cursor: not-allowed; }
    label { display: flex; flex-direction: column; gap: 0.5rem; font-weight: 600; }
    select, input[type=text] { padding: 0.6rem 0.8rem; font-size: 1rem; border: 1px solid #ced4da; border-radius: 6px; }
    .status { font-size: 0.95rem; color: #495057; min-height: 1.5rem; }
    audio { width: 100%; margin-top: 0.5rem; }
    .warning { font-size: 0.85rem; color: #c92a2a; }
    .result { background: rgba(255, 255, 255, 0.12); padding: 1.5rem; border-radius: 12px; margin-right: 1rem; margin-bottom: 1.5rem; }
    .result h2 { margin-top: 0; font-size: 1.4rem; }
    pre { white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; background: rgba(0, 0, 0, 0.35); padding: 1rem; border-radius: 8px; max-height: 40vh; overflow-y: auto; }
    @media (max-width: 960px) {
      main { flex-direction: column; }
      section { border-right: none; border-bottom: 1px solid #dee2e6; }
      .right { border-bottom: none; }
    }
  </style>
</head>
<body>
  <main>
    <section class='left'>
      <div>
        <h1>Enregistrer un message</h1>
        <p>Capturez un mémo vocal depuis votre navigateur puis envoyez-le à l'API pour transcription.</p>
        <p style='font-size:0.85rem;color:#495057;'>Fonctionne sur la plupart des navigateurs modernes. Safari iOS ne supporte pas encore l'enregistrement MediaRecorder.</p>
      </div>
      <div class='controls'>
        <button id='start-btn'>Démarrer</button>
        <button id='stop-btn' disabled>Stop</button>
        <button id='send-btn' class='secondary' disabled>Envoyer</button>
      </div>
      <label>
        Modèle
        <select id='model'>
          {{MODELS_OPTIONS}}
        </select>
      </label>
      <label>
        Langue imposée (optionnel)
        <input id='language' type='text' placeholder='fr, en, ...'>
      </label>
      <label style='flex-direction: row; align-items: center; gap: 0.5rem; font-weight: 500;'>
        <input id='vad' type='checkbox'>
        Filtre VAD (réduit les silences)
      </label>
      <label style='flex-direction: row; align-items: center; gap: 0.5rem; font-weight: 500;'>
        <input id='word-timestamps' type='checkbox'>
        Horodatages par mot
      </label>
      <div>
        <strong>Pré-écoute :</strong>
        <audio id='player' controls></audio>
      </div>
      <div class='status' id='status'>Prêt.</div>
      <div class='warning'>L'enregistrement reste local tant que vous n'appuyez pas sur « Envoyer ».</div>
    </section>
    <section class='right'>
      <div class='result'>
        <h2>Transcription</h2>
        <pre id='text-output'>Aucune donnée pour le moment.</pre>
      </div>
      <div class='result'>
        <h2>Métadonnées</h2>
        <pre id='meta-output'>{{}}
        </pre>
      </div>
    </section>
  </main>
  <script>
    let mediaRecorder = null;
    let audioChunks = [];
    let audioBlob = null;
    let mediaStream = null;

    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const sendBtn = document.getElementById('send-btn');
    const statusEl = document.getElementById('status');
    const player = document.getElementById('player');
    const textOutput = document.getElementById('text-output');
    const metaOutput = document.getElementById('meta-output');

    function setStatus(message) {
      statusEl.textContent = message;
    }

    async function ensureRecorder() {
      if (mediaRecorder) {
        return;
      }
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(mediaStream);
        mediaRecorder.addEventListener('dataavailable', (event) => {
          if (event.data && event.data.size > 0) {
            audioChunks.push(event.data);
          }
        });
        mediaRecorder.addEventListener('stop', () => {
          if (audioChunks.length) {
            audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            player.src = URL.createObjectURL(audioBlob);
            sendBtn.disabled = false;
            setStatus('Enregistrement terminé. Vous pouvez envoyer ou recommencer.');
          }
        });
      } catch (err) {
        console.error(err);
        setStatus("Accès au micro refusé ou indisponible.");
        startBtn.disabled = true;
      }
    }

    startBtn.addEventListener('click', async () => {
      await ensureRecorder();
      if (!mediaRecorder) {
        return;
      }
      audioChunks = [];
      audioBlob = null;
      player.removeAttribute('src');
      sendBtn.disabled = true;
      mediaRecorder.start();
      setStatus('Enregistrement en cours...');
      startBtn.disabled = true;
      stopBtn.disabled = false;
    });

    stopBtn.addEventListener('click', () => {
      if (!mediaRecorder || mediaRecorder.state !== 'recording') {
        return;
      }
      mediaRecorder.stop();
      stopBtn.disabled = true;
      startBtn.disabled = false;
      setStatus("Traitement de l'audio...");
    });

    sendBtn.addEventListener('click', async () => {
      if (!audioBlob) {
        setStatus('Aucun enregistrement à envoyer.');
        return;
      }
      setStatus('Envoi en cours...');
      sendBtn.disabled = true;

      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');
      formData.append('model', document.getElementById('model').value);
      const language = document.getElementById('language').value.trim();
      if (language) {
        formData.append('language', language);
      }
      if (document.getElementById('vad').checked) {
        formData.append('vad', 'true');
      }
      if (document.getElementById('word-timestamps').checked) {
        formData.append('word_timestamps', 'true');
      }

      try {
        const response = await fetch('/transcribe', {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) {
          const text = await response.text();
          throw new Error(`Erreur ${response.status}: ${text}`);
        }
        const data = await response.json();
        textOutput.textContent = data.text || '(réponse vide)';
        metaOutput.textContent = JSON.stringify(data, null, 2);
        setStatus('Transcription réussie.');
      } catch (err) {
        console.error(err);
        setStatus(`Erreur: ${err.message}`);
      } finally {
        sendBtn.disabled = false;
      }
    });

    window.addEventListener('beforeunload', () => {
      if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
      }
    });
  </script>
</body>
</html>
"""

    html = template.replace("{{MODELS_OPTIONS}}", models_options)
    return HTMLResponse(content=html)


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
