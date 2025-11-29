"""High-level inference helper."""

from __future__ import annotations

import logging

from app.models.base import BaseLLM, ChatMessage, GenerationConfig
from typing import Iterator, List, Optional
from app.models.gemma_hf import GemmaHF
from app.config import get_config


_LLM_INSTANCE: Optional[BaseLLM] = None
LOGGER = logging.getLogger(__name__)


def get_llm() -> BaseLLM:
    """Return the singleton language model instance."""
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        LOGGER.info("Initializing GemmaHF backend")
        _LLM_INSTANCE = GemmaHF()
    return _LLM_INSTANCE


def make_generation_config(
    *,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> GenerationConfig:
    """Build class:`GenerationConfig` using defaults where needed."""
    cfg = get_config()

    return GenerationConfig(
        max_new_tokens=max_new_tokens or cfg.default_max_new_tokens,
        temperature=temperature or cfg.default_temperature,
        top_p=top_p or cfg.default_top_p,
        top_k=top_k or cfg.default_top_k,
    )


def generate_reply(messages: List[ChatMessage], gen_config: GenerationConfig) -> str:
    """Generate a single non-streamed reply for the given message."""
    llm = get_llm()
    return llm.generate(messages, gen_config)


def stream_reply(
    messages: List[ChatMessage], gen_config: GenerationConfig
) -> Iterator[str]:
    """Stream a reply as an iterator of text chunks."""
    llm = get_llm()
    return llm.stream_generate(messages, gen_config)
