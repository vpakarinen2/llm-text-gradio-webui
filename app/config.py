"""Application configuration for LLM WebUI."""

from __future__ import annotations

import os

from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


def _get_bool_env(name: str, default: bool = False) -> bool:
    """Return a boolean value from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw_lower = raw.strip().lower()
    if raw_lower in {"1", "true", "yes", "on"}:
        return True
    if raw_lower in {"0", "false", "no", "off"}:
        return False
    return default


def _detect_device() -> str:
    """Detect a default device for running the model."""
    env_device = os.getenv("DEVICE")
    if env_device:
        return env_device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


@dataclass(frozen=True)
class AppConfig:
    """Immutable configuration object for the WebUI."""
    model_id: str = os.getenv("MODEL_ID", "google/gemma-3-4b-it")
    trust_remote_code: bool = _get_bool_env("TRUST_REMOTE_CODE", False)

    embed_model_id: str = os.getenv(
        "EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
    )

    device: str = _detect_device()
    torch_dtype: str = os.getenv("TORCH_DTYPE", "bfloat16")

    default_max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
    default_temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    default_top_p: float = float(os.getenv("TOP_P", "0.9"))
    default_top_k: int = int(os.getenv("TOP_K", "40"))

    host: str = os.getenv("HOST", "localhost")
    port: int = int(os.getenv("PORT", "7860"))
    share: bool = _get_bool_env("SHARE", False)
    queue_concurrency_limit: int = int(os.getenv("QUEUE_CONCURRENCY_LIMIT", "1"))


_CONFIG = AppConfig()


def get_config() -> AppConfig:
    """Return the application configuration instance."""
    return _CONFIG
