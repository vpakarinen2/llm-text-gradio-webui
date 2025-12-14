"""Hugging Face Transformers backend for Gemma model."""

from __future__ import annotations

import threading
import logging
import torch
import time

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from app.models.base import BaseLLM, ChatMessage, GenerationConfig
from typing import Dict, Iterator, List, Optional, Any
from app.config import AppConfig, get_config


LOGGER = logging.getLogger(__name__)


def _resolve_torch_dtype(name: str) -> Optional[torch.dtype]:
    """Map a string name to a ``torch.dtype``."""
    normalized = name.strip().lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    return None


class GemmaHF(BaseLLM):
    """Transformers-based backend for Gemma chat model."""
    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._config = config or get_config()
        self._device = torch.device(self._config.device)

        if self._device.type == "cuda":
            torch.cuda.init()
            torch.cuda.set_device(self._device.index or 0)

        LOGGER.info("Loading tokenizer for model_id=%s", self._config.model_id)

        self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_id)

        dtype = _resolve_torch_dtype(self._config.torch_dtype)
        model_kwargs: Dict[str, Any] = {}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        LOGGER.info(
            "Loading model for model_id=%s on device=%s",
            self._config.model_id,
            self._device,
        )

        if self._device.type == "cuda":
            model_kwargs["device_map"] = "cuda:0"
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self._config.model_id,
            **model_kwargs,
        )
        
        if "device_map" not in model_kwargs:
            self._model.to(self._device)
        
        self._model.eval()

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a prompt using the chat template."""
        payload = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

        prompt = self._tokenizer.apply_chat_template(
            payload,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Tokenize the prompt and move tensors to configured device."""
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    def generate(self, messages: List[ChatMessage], config: GenerationConfig) -> str:
        """Generate a single non-streamed response."""

        prompt = self._build_prompt(messages)
        inputs = self._encode_prompt(prompt)

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": True,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        LOGGER.info(
            "GemmaHF.generate: start (max_new_tokens=%d, temp=%.3f, top_p=%.3f, top_k=%d)",
            config.max_new_tokens,
            config.temperature,
            config.top_p,
            config.top_k,
        )

        start = time.perf_counter()

        if self._device.type == "cuda":
            torch.cuda.set_device(self._device.index or 0)

        with torch.no_grad():
            output = self._model.generate(**inputs, **generation_kwargs)[0]

        elapsed = time.perf_counter() - start

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output[prompt_length:]

        LOGGER.info(
            "GemmaHF.generate: done (elapsed=%.2fs, new_tokens=%d)",
            elapsed,
            int(generated_ids.shape[0]),
        )

        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    def stream_generate(
        self, messages: List[ChatMessage], config: GenerationConfig
    ) -> Iterator[str]:
        """Generate a response as a stream of text chunks."""
        LOGGER.info(
            "GemmaHF.stream_generate: start (max_new_tokens=%d)",
            config.max_new_tokens,
        )
        
        prompt = self._build_prompt(messages)
        inputs = self._encode_prompt(prompt)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": True,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }

        def _worker() -> None:
            try:
                if self._device.type == "cuda":
                    torch.cuda.set_device(self._device.index or 0)
                with torch.no_grad():
                    self._model.generate(**inputs, **generation_kwargs)
            except Exception:
                LOGGER.exception("Error during streaming generation")

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        for text in streamer:
            yield text
        
        LOGGER.info("GemmaHF.stream_generate: done")
