"""Embedding client.

Talks to OpenRouter's OpenAI-compatible /embeddings endpoint. Caches
every embedding to disk keyed by (model, sha256 of text) so a re-run
costs nothing. Has a --mock fallback that produces deterministic
pseudo-embeddings without touching the network, so the pipeline runs
end-to-end even if the demo network is bad.

The mock path still counts tokens the way the real path would, so the
cost panel numbers are the numbers we WOULD have paid.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
import tiktoken

from .cost import UsageLedger


CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "embeddings"
DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "openai/text-embedding-3-small")
DEFAULT_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
EMBED_DIM = 1536  # text-embedding-3-small default dimension


# tiktoken doesn't know OpenRouter model names, so use the underlying
# OpenAI tokenizer directly for token counting.
_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _cache_path(model: str, text: str) -> Path:
    key = hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _load_cached(model: str, text: str) -> np.ndarray | None:
    p = _cache_path(model, text)
    if not p.exists():
        return None
    with p.open() as f:
        return np.array(json.load(f)["embedding"], dtype=np.float32)


def _save_cache(model: str, text: str, vec: np.ndarray) -> None:
    p = _cache_path(model, text)
    with p.open("w") as f:
        json.dump({"model": model, "embedding": vec.tolist()}, f)


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#\-]*")


def _mock_embedding(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic pseudo-embedding via hashed bag-of-tokens.

    Each token in the text is hashed to a bucket in [0, dim). Texts
    that share vocabulary end up with overlapping non-zero dimensions
    and therefore non-trivial cosine similarity — enough for the
    notebook's constraint-break demo to reproduce its intended
    ordering without a network call.

    This is not a semantic embedding. It has no notion of synonymy,
    word order, or negation. It's a plausibility-preserving fallback
    for when the OpenRouter endpoint is unreachable during a live
    demo. The notebook flags this in the mock-run outputs.
    """
    v = np.zeros(dim, dtype=np.float32)
    for tok in _WORD_RE.findall(text.lower()):
        h = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:4], "big")
        v[h % dim] += 1.0
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v


def _call_openrouter(texts: list[str], model: str, api_key: str, base_url: str) -> list[np.ndarray]:
    r = requests.post(
        f"{base_url.rstrip('/')}/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/local/jd-matcher",
            "X-Title": "jd-matcher",
        },
        json={"model": model, "input": texts},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()["data"]
    return [np.array(d["embedding"], dtype=np.float32) for d in data]


def embed_many(
    texts: Iterable[str],
    ledger: UsageLedger,
    *,
    model: str = DEFAULT_MODEL,
    mock: bool = False,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> np.ndarray:
    """Embed a list of texts. Returns a stacked (N, dim) float32 matrix.

    - Cache hits do not increment tokens_embedded (we did not pay again).
    - Cache misses do increment tokens_embedded regardless of mock/real,
      so the cost panel reports what a fresh run WOULD have cost.
    """
    texts = list(texts)
    vecs: list[np.ndarray | None] = [None] * len(texts)
    misses: list[tuple[int, str]] = []

    for i, t in enumerate(texts):
        cached = _load_cached(model, t)
        if cached is not None:
            vecs[i] = cached
            ledger.embeddings_from_cache += 1
        else:
            misses.append((i, t))

    if misses:
        miss_texts = [t for _, t in misses]
        ledger.embeddings_computed += len(miss_texts)
        ledger.tokens_embedded += sum(count_tokens(t) for t in miss_texts)

        if mock:
            new_vecs = [_mock_embedding(t) for t in miss_texts]
        else:
            if not api_key:
                raise RuntimeError(
                    "No OPENROUTER_API_KEY provided and mock=False. "
                    "Set the env var or pass mock=True."
                )
            new_vecs = _call_openrouter(miss_texts, model, api_key, base_url)

        for (i, t), v in zip(misses, new_vecs):
            vecs[i] = v
            _save_cache(model, t, v)

    return np.stack(vecs)  # type: ignore[arg-type]


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. a: (N, d), b: (M, d) -> (N, M)."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T
