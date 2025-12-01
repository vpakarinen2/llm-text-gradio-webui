from __future__ import annotations

import numpy as np
import os

from sentence_transformers import SentenceTransformer
from app.config import get_config
from typing import List, Tuple


_EMBED_MODEL: SentenceTransformer | None = None
_DOC_EMBEDDINGS: np.ndarray | None = None
_DOC_TEXTS: List[str] = []

_EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")


def get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        cfg = get_config()
        device = cfg.device if cfg.device in {"cuda", "cpu"} else "cpu"
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_ID, device=device)
    return _EMBED_MODEL


def reset_index() -> None:
    global _DOC_EMBEDDINGS, _DOC_TEXTS
    _DOC_EMBEDDINGS = None
    _DOC_TEXTS = []


def index_texts(texts: List[str]) -> int:
    """Index a batch of raw document texts."""
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return 0

    model = get_embed_model()
    embeddings = model.encode(
        cleaned,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    global _DOC_EMBEDDINGS, _DOC_TEXTS
    if _DOC_EMBEDDINGS is None:
        _DOC_EMBEDDINGS = embeddings
        _DOC_TEXTS = list(cleaned)
    else:
        _DOC_EMBEDDINGS = np.concatenate([_DOC_EMBEDDINGS, embeddings], axis=0)
        _DOC_TEXTS.extend(cleaned)

    return len(cleaned)


def is_index_empty() -> bool:
    return _DOC_EMBEDDINGS is None or _DOC_EMBEDDINGS.shape[0] == 0


def retrieve(query: str, top_k: int = 4) -> List[Tuple[str, float]]:
    """Return top-k documents and similarity scores."""
    if is_index_empty():
        return []

    model = get_embed_model()
    query = query.strip()
    if not query:
        return []

    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    scores = _DOC_EMBEDDINGS @ query_emb
    k = min(max(top_k, 1), _DOC_EMBEDDINGS.shape[0])
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [(_DOC_TEXTS[i], float(scores[i])) for i in top_idx]
