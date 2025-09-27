#!/usr/bin/env python3
import argparse
import pathlib
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from faster_whisper import WhisperModel


@dataclass
class TranscriptionResult:
    lines: List[str]
    segments: List[Dict[str, Any]]
    info: Dict[str, Any]
    word_count: int
    char_count: int

    @property
    def segment_count(self) -> int:
        return len(self.lines)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def resolve_device(device_option: str) -> str:
    if device_option != "auto":
        return device_option
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_compute_type(device: str, compute_type_option: Optional[str]) -> str:
    if compute_type_option:
        return compute_type_option
    return "float32" if device == "cpu" else "float16"


def create_model(
    model_path: Union[str, pathlib.Path],
    *,
    device_option: str = "auto",
    compute_type_option: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> Tuple[WhisperModel, str, str]:
    if device is None:
        device = resolve_device(device_option)
    if compute_type is None:
        compute_type = resolve_compute_type(device, compute_type_option)
    model = WhisperModel(str(model_path), device=device, compute_type=compute_type)
    return model, device, compute_type


def transcribe_audio(
    *,
    model: WhisperModel,
    audio_path: Union[str, pathlib.Path],
    language: Optional[str] = None,
    vad: bool = False,
    word_timestamps: bool = False,
    beam_size: int = 5,
    temperature: float = 0.0,
    best_of: int = 5,
) -> TranscriptionResult:
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        vad_filter=vad,
        vad_parameters=dict(min_silence_duration_ms=500) if vad else None,
        word_timestamps=word_timestamps,
        beam_size=beam_size,
        temperature=temperature,
        best_of=best_of,
    )

    lines: List[str] = []
    segments: List[Dict[str, Any]] = []
    word_count = 0
    char_count = 0

    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue

        lines.append(text)
        word_count += len(text.split())
        char_count += len(text)

        payload: Dict[str, Any] = {
            "start": seg.start,
            "end": seg.end,
            "text": text,
        }

        if word_timestamps and getattr(seg, "words", None):
            payload["words"] = [
                {"start": w.start, "end": w.end, "word": w.word}
                for w in seg.words
                if getattr(w, "word", "").strip()
            ]

        segments.append(payload)

    info_dict = {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
    }

    return TranscriptionResult(
        lines=lines,
        segments=segments,
        info=info_dict,
        word_count=word_count,
        char_count=char_count,
    )


def write_transcription(result: TranscriptionResult, destination: pathlib.Path) -> None:
    if destination.parent and not destination.parent.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)

    text_content = result.text
    if result.lines:
        text_content += "\n"

    destination.write_text(text_content, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcrire un fichier audio/vidéo en texte (local, offline)."
    )
    parser.add_argument("audio", help="Chemin du fichier (mp3, wav, m4a, mp4, etc.)")
    parser.add_argument(
        "-o",
        "--output",
        help="Fichier texte de sortie (défaut: <audio>.txt)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="./models/whisper-medium",
        help="Chemin d’un dossier modèle OU nom HuggingFace (défaut: ./models/whisper-medium)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Forcer l’appareil: auto/cpu/cuda (défaut: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Exemples: int8, int8_float16, float16, float32 (défaut: auto selon device)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Code langue (ex: fr, en). Laisser vide pour auto-détection.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Timestamps au mot (plus lent).",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Filtrage VAD pour réduire le bruit/les silences.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    audio_path = pathlib.Path(args.audio)
    if not audio_path.exists():
        sys.exit(f"Fichier introuvable: {audio_path}")

    out_path = pathlib.Path(args.output) if args.output else audio_path.with_suffix(".txt")

    device = resolve_device(args.device)
    compute_type = resolve_compute_type(device, args.compute_type)

    start_ts = time.time()
    start_dt = datetime.now()
    print(
        f"[{start_dt:%Y-%m-%d %H:%M:%S}] Début transcription de '{audio_path}' "
        f"avec le modèle '{args.model}' sur {device} ({compute_type})."
    )

    model, _, _ = create_model(
        args.model,
        device=device,
        compute_type=compute_type,
    )

    result = transcribe_audio(
        model=model,
        audio_path=audio_path,
        language=args.language,
        vad=args.vad,
        word_timestamps=args.word_timestamps,
    )

    write_transcription(result, out_path)

    duration = time.time() - start_ts
    end_dt = datetime.now()
    print(
        f"[{end_dt:%Y-%m-%d %H:%M:%S}] Fin transcription: {out_path} | "
        f"segments={result.segment_count} | mots={result.word_count} | "
        f"caractères={result.char_count} | durée={duration:.1f}s"
    )


if __name__ == "__main__":
    main()
