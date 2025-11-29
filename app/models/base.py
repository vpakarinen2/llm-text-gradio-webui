"""Base abstraction for language model backend."""

from __future__ import annotations

from typing import Iterator, List, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass


Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    """Single chat message exchanged with the model."""
    role: Role
    content: str


@dataclass
class GenerationConfig:
    """Configuration for a single text generation request."""
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int


class BaseLLM(ABC):
    """Abstract base class for language model backend."""
    @abstractmethod
    def generate(self, messages: List[ChatMessage], config: GenerationConfig) -> str:
        """Generate a single response for the given conversation."""
        raise NotImplementedError

    @abstractmethod
    def stream_generate(
        self, messages: List[ChatMessage], config: GenerationConfig
    ) -> Iterator[str]:
        """Generate a response as an iterator of text chunks."""
        raise NotImplementedError
