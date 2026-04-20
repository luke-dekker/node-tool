"""Text → vector backends for the Memory subsystem.

Default: `SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")`. Reuses
the torch already installed for training. Model is lazy-loaded on first
`embed` call; constructing the embedder is free.

`HF_HUB_OFFLINE=1` is set at load time to suppress network traffic after
the first (cached) download.
"""
from __future__ import annotations
import os
from typing import ClassVar

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_KNOWN_DIMS = {
    "all-MiniLM-L6-v2":   384,
    "all-mpnet-base-v2":  768,
    "bge-small-en-v1.5":  384,
    "bge-base-en-v1.5":   768,
}


class SentenceTransformerEmbedder:
    """Wraps `sentence_transformers.SentenceTransformer`.

    Batch-encodes inputs in a single `encode()` call. Vectors are
    L2-normalized so cosine-similarity search is a plain inner product.
    """
    capabilities: ClassVar[set[str]] = {"batch", "normalize"}

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name or _DEFAULT_MODEL
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            # Enforce offline after first download — embeddings must not reach
            # out to HF every run.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer  # deferred
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        if self._model is not None:
            return int(self._model.get_sentence_embedding_dimension())
        return _KNOWN_DIMS.get(self.model_name, 0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vecs = model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vecs]


class HashEmbedder:
    """Deterministic no-dep embedder — feature-hashes token counts into a
    fixed-dim vector, then L2-normalizes.

    Used as a fallback when sentence-transformers is absent, and as the
    default in tests so the suite doesn't need torch. Not semantic — good
    enough for keyword-overlap retrieval.
    """
    capabilities: ClassVar[set[str]] = {"batch", "normalize", "no_deps"}

    def __init__(self, dim: int = 256):
        self.model_name = f"hash-{dim}"
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        import math
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self._dim
            for tok in _tokenize(t):
                idx = hash(("tok", tok)) % self._dim
                sign = 1.0 if (hash(("sgn", tok)) & 1) else -1.0
                v[idx] += sign
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


def build_embedder(model_name: str, device: str | None = None):
    """Build the configured embedder; fall back to HashEmbedder if
    sentence-transformers is unavailable AND the graph isn't demanding it.

    The explicit `hash-*` model names always resolve to HashEmbedder.
    """
    if model_name and model_name.startswith("hash"):
        try:
            dim = int(model_name.split("-", 1)[1])
        except (IndexError, ValueError):
            dim = 256
        return HashEmbedder(dim=dim)
    return SentenceTransformerEmbedder(
        model_name=model_name or _DEFAULT_MODEL, device=device,
    )
