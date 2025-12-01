"""Callback helper for the Gradio chat interface."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from app.inference import make_generation_config, generate_reply, stream_reply
from typing import Iterator, List, Optional, Tuple
from app.models.base import ChatMessage
from app import rag


LOGGER = logging.getLogger(__name__)
ChatHistory = List[Tuple[str, Optional[str]]]


def build_messages(
    history: ChatHistory,
    new_user_message: str,
    system_prompt: Optional[str] = None,
) -> List[ChatMessage]:
    """Convert chat history and a new user message."""
    messages: List[ChatMessage] = []

    if system_prompt:
        text = system_prompt.strip()
        if text:
            LOGGER.info(
                "build_messages: applying system_prompt (len=%d): %s",
                len(text),
                text[:200].replace("\n", " "),
            )
            messages.append(ChatMessage(role="system", content=text))
        else:
            LOGGER.info("build_messages: system_prompt provided but empty after strip")
    else:
        LOGGER.info("build_messages: no system_prompt provided")

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
    """Main callback for handling a user message."""
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


def index_rag_files(files: List[Path]) -> str:
    """Index uploaded text files for RAG."""

    if not files:
        return "No files provided."

    texts: List[str] = []
    for raw in files:
        path = Path(raw)
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            texts.append(content)
        except Exception as exc:
            LOGGER.exception("index_rag_files: failed to read %s", path)
            return f"Error reading file {path.name}: {exc}"

    count = rag.index_texts(texts)
    return f"Indexed {count} document(s) for RAG."


def handle_rag_message(
    user_message: str,
    history: ChatHistory,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    top_k: int,
    system_prompt: str,
    top_k_docs: int,
) -> Tuple[ChatHistory, str, str]:
    """RAG-enabled chat callback."""

    history = list(history) if history else []
    user_message = (user_message or "").strip()
    if not user_message:
        return history, "", ""

    if rag.is_index_empty():
        status = "No documents indexed. Upload and index files first."
        return history, "", status

    retrieved = rag.retrieve(user_message, top_k_docs)
    if not retrieved:
        status = "No relevant documents found for the query."
        return history, "", status

    max_total_chars = 8000
    per_doc_limit = max_total_chars // len(retrieved)
    if per_doc_limit < 1000:
        per_doc_limit = 1000

    context_blocks = []
    for idx, (text, score) in enumerate(retrieved, start=1):
        snippet = text.strip()
        if len(snippet) > per_doc_limit:
            snippet = snippet[:per_doc_limit]
        context_blocks.append(f"[{idx}] (score={score:.3f})\n{snippet}")
    context = "\n\n".join(context_blocks)

    base_system = (system_prompt or "").strip()
    augmented_system = (
        (base_system + "\n\n" if base_system else "")
        + "Use ONLY the following context to answer the user's question. "
        + "If the answer is not in the context, say you don't know.\n\n"
        + f"Context:\n{context}"
    )

    LOGGER.info(
        "handle_rag_message: using %d context docs", len(retrieved)
    )

    messages = build_messages(history, user_message, augmented_system)

    gen_config = make_generation_config(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    reply = generate_reply(messages, gen_config).strip()
    history.append((user_message, reply if reply else "..."))

    status = f"Retrieved {len(retrieved)} document(s) for this answer."
    return history, "", status
