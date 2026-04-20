"""Vector-store protocol + shared dataclasses for the Memory subsystem.

Importing this module must not pull qdrant_client / chromadb / lancedb.
Backend adapters defer their heavy imports to first method call.

The `MemoryRef` dataclass is the value carried on `MEMORY_REF` ports — a
string handle, not a live client. Graphs that reference `MemoryRef` are
fully JSON-serializable. `open_store(ref)` resolves it back to a live
`VectorStore` inside `execute()`.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable
import uuid


@dataclass
class Document:
    """A single chunk. `id` is stable across re-runs if the caller supplies one;
    otherwise we derive a content-addressed id so repeat upserts are idempotent.
    """
    text: str
    metadata: dict = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = stable_id(self.text, self.metadata)

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "metadata": dict(self.metadata)}


@dataclass
class Embedding:
    """Dense vector + its dim. Distinct type from TENSOR.

    `model` records which embedder produced this vector — `MemoryStoreNode`
    refuses to mix vectors from different models in one collection.
    """
    vector: list[float]
    dim: int = 0
    model: str = ""

    def __post_init__(self) -> None:
        if self.dim == 0:
            self.dim = len(self.vector)
        elif self.dim != len(self.vector):
            raise ValueError(
                f"Embedding dim mismatch: declared {self.dim}, got {len(self.vector)}"
            )


@dataclass
class MemoryRef:
    """Serializable handle to a vector store.

    The same ref can be passed to `RetrieverNode` in the same graph or a
    different one — the backend is opened lazily on first use.
    """
    backend: str = "qdrant"         # qdrant | chroma | lance
    path:    str = "./qdrant_data"
    collection: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class VectorStore(Protocol):
    """Minimum surface every backend implements.

    Upsert is idempotent on `id`. Query returns `(document, score)` pairs,
    score-descending. `where` is an optional backend-normalized filter dict
    (Qdrant's Filter DSL is the reference; other backends translate).
    """
    capabilities: ClassVar[set[str]]   # subset of {filter, persistent, batch}

    def upsert(
        self, docs: list[Document], vectors: list[list[float]],
    ) -> int: ...

    def query(
        self, query_vector: list[float], *, k: int = 5,
        where: dict | None = None,
    ) -> list[tuple[Document, float]]: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


def stable_id(text: str, metadata: dict | None = None) -> str:
    """Deterministic UUIDv5 over text + normalized metadata.

    Used when the caller doesn't supply an id. Two DocumentLoaderNode runs on
    the same file produce the same chunk ids, so MemoryStoreNode.upsert is a
    no-op on the second run.
    """
    import hashlib
    h = hashlib.sha1()
    h.update((text or "").encode("utf-8"))
    if metadata:
        for k in sorted(metadata):
            v = metadata[k]
            h.update(f"\x1f{k}={v!r}".encode("utf-8"))
    return str(uuid.UUID(h.hexdigest()[:32]))


def open_store(ref: MemoryRef) -> VectorStore:
    """Resolve a MemoryRef to a live VectorStore. Heavy import happens here."""
    if ref.backend == "qdrant":
        from plugins.agents._memory.qdrant_backend import QdrantBackend
        return QdrantBackend(path=ref.path, collection=ref.collection)
    raise NotImplementedError(
        f"MemoryStore backend {ref.backend!r} not implemented (qdrant only in v1)"
    )
