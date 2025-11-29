"""Callback helper for the Gradio chat interface."""

from __future__ import annotations

import logging
import time

from app.inference import make_generation_config, generate_reply, stream_reply
from typing import List, Tuple, Iterator, Optional
from app.models.base import ChatMessage


LOGGER = logging.getLogger(__name__)
ChatHistory = List[Tuple[str, Optional[str]]]


def build_messages(
    history: ChatHistory,
    new_user_message: str,
    system_prompt: Optional[str] = None,
) -> List[ChatMessage]:
    """Convert chat history (tuple format) and a new user message."""
    messages: List[ChatMessage] = []

    if system_prompt:
        text = system_prompt.strip()
        if text:
            messages.append(ChatMessage(role="system", content=text))

    for user_msg, bot_msg in history or []:
        if user_msg:
            messages.append(ChatMessage(role="user", content=user_msg.strip()))
        if bot_msg:
            messages.append(ChatMessage(role="assistant", content=bot_msg.strip()))

    user_text = new_user_message.strip()
    if user_text:
        messages.append(ChatMessage(role="user", content=user_text))

    return messages


def handle_user_message(
    user_message: str,
    history: ChatHistory,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    top_k: int,
    system_prompt: str,
) -> Tuple[ChatHistory, str]:
    """Main callback for handling a user message (non-streaming)."""
    started = time.perf_counter()

    history = list(history) if history else []
    user_message = (user_message or "").strip()
    if not user_message:
        LOGGER.info("handle_user_message: empty input, nothing to do")
        return history, ""

    LOGGER.info(
        "handle_user_message: start (len_history=%d, max_new_tokens=%d, temp=%.3f, top_p=%.3f, top_k=%d)",
        len(history),
        max_new_tokens,
        temperature,
        top_p,
        top_k,
    )

    messages = build_messages(history, user_message, system_prompt)

    gen_config = make_generation_config(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    reply = generate_reply(messages, gen_config)
    gen_elapsed = time.perf_counter() - started
    LOGGER.info(
        "handle_user_message: generate_reply finished (len_reply=%d, elapsed=%.2fs)",
        len(reply),
        gen_elapsed,
    )

    reply = reply.strip()
    history.append((user_message, reply if reply else "..."))

    total_elapsed = time.perf_counter() - started
    LOGGER.info("handle_user_message: end (total_elapsed=%.2fs)", total_elapsed)

    return history, ""


def clear_history() -> Tuple[ChatHistory, str]:
    """Reset the chat history and input box."""
    return [], ""


def handle_user_message_streaming(
    user_message: str,
    history: ChatHistory,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    top_k: int,
    system_prompt: str,
) -> Iterator[Tuple[ChatHistory, str]]:
    """Streaming callback that yields partial responses."""
    history = list(history) if history else []
    user_message = (user_message or "").strip()
    if not user_message:
        yield history, ""
        return

    LOGGER.info(
        "handle_user_message_streaming: start (len_history=%d, max_new_tokens=%d)",
        len(history),
        max_new_tokens,
    )

    messages = build_messages(history, user_message, system_prompt)

    gen_config = make_generation_config(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    history.append((user_message, ""))
    yield history, ""
    
    accumulated = ""
    last_yield_time = time.perf_counter()
    MIN_YIELD_INTERVAL = 0.1
    
    for chunk in stream_reply(messages, gen_config):
        accumulated += chunk
        
        now = time.perf_counter()
        if now - last_yield_time >= MIN_YIELD_INTERVAL:
            history[-1] = (user_message, accumulated)
            yield history, ""
            last_yield_time = now

    history[-1] = (user_message, accumulated.strip())
    LOGGER.info("handle_user_message_streaming: done (len_reply=%d)", len(accumulated))
    yield history, ""
