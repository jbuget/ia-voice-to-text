from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    model_root: Path = Path(os.getenv("TRANSCRIBE_MODEL_ROOT", "./models")).resolve()
    default_model_name: str = os.getenv("TRANSCRIBE_DEFAULT_MODEL", "whisper-medium")
    default_model_path: Path = Path(
        os.getenv(
            "TRANSCRIBE_MODEL",
            str(Path(os.getenv("TRANSCRIBE_MODEL_ROOT", "./models")).resolve() / os.getenv("TRANSCRIBE_DEFAULT_MODEL", "whisper-medium")),
        )
    ).resolve()
    device_option: str = os.getenv("TRANSCRIBE_DEVICE", "auto")
    compute_type_option: str | None = os.getenv("TRANSCRIBE_COMPUTE_TYPE")
    forward_url: str = os.getenv("TRANSCRIBE_FORWARD_URL", "")


__all__ = ["Settings"]
