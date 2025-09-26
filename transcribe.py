#!/usr/bin/env python3
import argparse, pathlib, sys, time
from datetime import datetime
from faster_whisper import WhisperModel

def main():
    p = argparse.ArgumentParser(description="Transcrire un fichier audio/vidéo en texte (local, offline).")
    p.add_argument("audio", help="Chemin du fichier (mp3, wav, m4a, mp4, etc.)")
    p.add_argument("-o", "--output", help="Fichier texte de sortie (défaut: <audio>.txt)")
    p.add_argument("-m", "--model", default="./models/whisper-large-v3",
                   help="Chemin d’un dossier modèle OU nom HuggingFace (défaut: ./models/whisper-large-v3)")
    p.add_argument("--device", default="auto", choices=["auto","cpu","cuda"],
                   help="Forcer l’appareil: auto/cpu/cuda (défaut: auto)")
    p.add_argument("--compute-type", default=None,
                   help="Exemples: int8, int8_float16, float16, float32 (défaut: auto selon device)")
    p.add_argument("--language", default=None,
                   help="Code langue (ex: fr, en). Laisser vide pour auto-détection.")
    p.add_argument("--translate-to-en", action="store_true",
                   help="Traduire vers l’anglais (au lieu de transcrire).")
    p.add_argument("--word-timestamps", action="store_true",
                   help="Timestamps au mot (plus lent).")
    p.add_argument("--vad", action="store_true",
                   help="Filtrage VAD pour réduire le bruit/les silences.")
    args = p.parse_args()

    audio_path = pathlib.Path(args.audio)
    if not audio_path.exists():
        sys.exit(f"Fichier introuvable: {audio_path}")

    out_path = pathlib.Path(args.output) if args.output else audio_path.with_suffix(".txt")

    # Résolution automatique du device si demandé
    device = args.device
    if device == "auto":
        try:
            import torch  # juste pour tester CUDA dispo si installé
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # Charge le modèle (chemin local OU nom de repo HF). Hors-ligne si chemin local.
    if args.compute_type is not None:
        compute_type = args.compute_type
    else:
        compute_type = "float32" if device == "cpu" else "float16"

    start_ts = time.time()
    start_dt = datetime.now()
    print(
        f"[{start_dt:%Y-%m-%d %H:%M:%S}] Début transcription de '{audio_path}' "
        f"avec le modèle '{args.model}' sur {device} ({compute_type})."
    )

    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        str(audio_path),
        language=args.language,
        task="translate" if args.translate_to_en else "transcribe",
        vad_filter=args.vad,
        vad_parameters=dict(min_silence_duration_ms=500) if args.vad else None,
        word_timestamps=args.word_timestamps,
        beam_size=5,
        temperature=0.0,
        best_of=5,
    )

    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    total_segments = 0
    total_words = 0
    total_chars = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            f.write(f"{text}\n")
            total_segments += 1
            total_words += len(text.split())
            total_chars += len(text)

    duration = time.time() - start_ts
    end_dt = datetime.now()
    print(
        f"[{end_dt:%Y-%m-%d %H:%M:%S}] Fin transcription: {out_path} | "
        f"segments={total_segments} | mots={total_words} | caractères={total_chars} | "
        f"durée={duration:.1f}s"
    )

if __name__ == "__main__":
    main()
