"""Qdrant backend for VectorStore. Defers `import qdrant_client` to first call.

Local mode (`QdrantClient(path=…)`) runs Qdrant's indexing logic in-process,
no separate server. The same client object reaches a remote `:6333` if
`path` is passed as a URL instead — zero code change in the node.
"""
from __future__ import annotations
from typing import Any, ClassVar

from plugins.agents._memory.store_protocol import Document


class QdrantBackend:
    """In-process Qdrant persistent store.

    Collection is created on first `upsert` once the vector dim is known.
    Upserts are idempotent on Document.id.
    """
    capabilities: ClassVar[set[str]] = {"filter", "persistent", "batch"}

    def __init__(self, path: str = "./qdrant_data", collection: str = "default"):
        self.path = path
        self.collection = collection
        self._client = None
        self._dim: int | None = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient  # deferred
            if isinstance(self.path, str) and (
                self.path.startswith("http://") or self.path.startswith("https://")
            ):
                self._client = QdrantClient(url=self.path)
            else:
                self._client = QdrantClient(path=self.path)
        return self._client

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.http import models as qm  # deferred
        client = self._get_client()
        existing = {c.name for c in client.get_collections().collections}
        if self.collection in existing:
            info = client.get_collection(self.collection)
            existing_dim = info.config.params.vectors.size
            if existing_dim != dim:
                raise ValueError(
                    f"Collection {self.collection!r} has dim {existing_dim}, "
                    f"got vectors with dim {dim}. Use a different collection "
                    "or delete the existing one."
                )
            self._dim = existing_dim
            return
        client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        self._dim = dim

    def upsert(
        self, docs: list[Document], vectors: list[list[float]],
    ) -> int:
        if not docs:
            return 0
        if len(docs) != len(vectors):
            raise ValueError(
                f"upsert len mismatch: {len(docs)} docs vs {len(vectors)} vectors"
            )
        from qdrant_client.http import models as qm  # deferred
        dim = len(vectors[0])
        self._ensure_collection(dim)
        client = self._get_client()
        points = [
            qm.PointStruct(
                id=d.id,
                vector=list(v),
                payload={"text": d.text, "metadata": d.metadata},
            )
            for d, v in zip(docs, vectors)
        ]
        client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def query(
        self, query_vector: list[float], *, k: int = 5,
        where: dict | None = None,
    ) -> list[tuple[Document, float]]:
        client = self._get_client()
        flt = _build_filter(where) if where else None
        # qdrant-client 1.10+ exposes query_points; older exposed search.
        if hasattr(client, "query_points"):
            resp = client.query_points(
                collection_name=self.collection,
                query=list(query_vector),
                limit=int(k),
                query_filter=flt,
                with_payload=True,
            )
            hits = resp.points
        else:
            hits = client.search(
                collection_name=self.collection,
                query_vector=list(query_vector),
                limit=int(k),
                query_filter=flt,
            )
        out: list[tuple[Document, float]] = []
        for h in hits:
            payload = h.payload or {}
            doc = Document(
                id=str(h.id),
                text=str(payload.get("text", "")),
                metadata=dict(payload.get("metadata", {}) or {}),
            )
            out.append((doc, float(h.score)))
        return out

    def count(self) -> int:
        try:
            client = self._get_client()
            return int(client.count(collection_name=self.collection, exact=True).count)
        except Exception:
            return 0

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None


def _build_filter(where: dict) -> Any:
    """Translate a flat `{key: value}` or `{key: {"$in": [...]}}` dict into a
    Qdrant Filter. Values become `must` conditions on `metadata.<key>`.
    """
    from qdrant_client.http import models as qm  # deferred
    conditions = []
    for key, value in where.items():
        field = f"metadata.{key}"
        if isinstance(value, dict) and "$in" in value:
            conditions.append(qm.FieldCondition(
                key=field, match=qm.MatchAny(any=list(value["$in"])),
            ))
        else:
            conditions.append(qm.FieldCondition(
                key=field, match=qm.MatchValue(value=value),
            ))
    return qm.Filter(must=conditions)
