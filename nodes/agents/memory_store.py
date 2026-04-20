"""MemoryStoreNode — upsert DOCUMENT[]+EMBEDDING[] into a vector store.

Output is a `MemoryRef` — a serializable string handle `{backend, path,
collection}`, not a live client. RetrieverNode resolves the ref back to
a live store on its own side. This keeps the graph JSON-serializable and
lets the same ref cross RPC boundaries.

Upsert is idempotent on `Document.id` — re-running a graph against an
existing store updates in place instead of duplicating.
"""
from __future__ import annotations
from typing import Any

from core.node import BaseNode, PortType


class MemoryStoreNode(BaseNode):
    type_name   = "ag_memory_store"
    label       = "Memory Store"
    category    = "Agents"
    subcategory = "Memory"
    description = ("Persist DOCUMENT[]+EMBEDDING[] pairs into a vector store "
                   "(Qdrant local by default). Emits a MEMORY_REF handle.")

    _BACKENDS = ("qdrant",)   # chroma/lance land in a later pass

    def _setup_ports(self) -> None:
        self.add_input("documents", "DOCUMENT", default=None,
                       description="DOCUMENT[] to upsert")
        self.add_input("embeddings", "EMBEDDING", default=None,
                       description="EMBEDDING[] aligned with documents")
        self.add_input("backend", PortType.STRING, default="qdrant",
                       description="qdrant (default). chroma/lance reserved.")
        self.add_input("path", PortType.STRING, default="./qdrant_data",
                       description="Local directory, or http(s):// URL for a remote Qdrant")
        self.add_input("collection", PortType.STRING, default="default",
                       description="Collection name within the store")
        self.add_output("store_ref", "MEMORY_REF")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._memory.store_protocol import (  # deferred
            Document, Embedding, MemoryRef, open_store,
        )

        backend = (inputs.get("backend") or "qdrant").strip().lower()
        if backend not in self._BACKENDS:
            raise RuntimeError(
                f"MemoryStoreNode: backend {backend!r} not supported "
                f"(supported: {', '.join(self._BACKENDS)})"
            )
        path = (inputs.get("path") or "./qdrant_data").strip() or "./qdrant_data"
        collection = (inputs.get("collection") or "default").strip() or "default"

        docs_in = inputs.get("documents")
        embs_in = inputs.get("embeddings")
        docs = _normalize_list(docs_in, Document)
        embs = _normalize_list(embs_in, Embedding)

        ref = MemoryRef(backend=backend, path=path, collection=collection)

        if not docs:
            return {"store_ref": ref}
        if len(docs) != len(embs):
            raise RuntimeError(
                f"MemoryStoreNode: {len(docs)} documents vs {len(embs)} embeddings"
            )
        dims = {e.dim for e in embs}
        if len(dims) != 1:
            raise RuntimeError(
                f"MemoryStoreNode: mixed embedding dims {sorted(dims)} — pass "
                "embeddings from a single embedder."
            )
        store = open_store(ref)
        try:
            store.upsert(docs, [e.vector for e in embs])
        finally:
            store.close()
        return {"store_ref": ref}

    def export(self, iv, ov):
        docs_in = iv.get("documents")  or "[]"
        embs_in = iv.get("embeddings") or "[]"
        backend = (self.inputs["backend"].default_value or "qdrant").strip()
        path = (self.inputs["path"].default_value or "./qdrant_data").strip()
        collection = (self.inputs["collection"].default_value or "default").strip()

        ref_var = ov.get("store_ref", f"_ref_{self.safe_id}")
        lines: list[str] = [
            f"_docs_in = list({docs_in} or [])",
            f"_embs_in = list({embs_in} or [])",
            f"{ref_var} = {{'backend': {backend!r}, 'path': {path!r}, "
            f"'collection': {collection!r}}}",
            "if _docs_in and _embs_in:",
            "    assert len(_docs_in) == len(_embs_in), "
            "f'{len(_docs_in)} docs vs {len(_embs_in)} embeddings'",
            "    _dim = _embs_in[0]['dim']",
            f"    _cli = QdrantClient(path={path!r})",
            "    _existing = {c.name for c in _cli.get_collections().collections}",
            f"    if {collection!r} not in _existing:",
            f"        _cli.create_collection({collection!r}, "
            "vectors_config=VectorParams(size=_dim, distance=Distance.COSINE))",
            "    _points = [PointStruct(id=d['id'], vector=e['vector'], "
            "payload={'text': d['text'], 'metadata': d['metadata']}) "
            "for d, e in zip(_docs_in, _embs_in)]",
            f"    _cli.upsert(collection_name={collection!r}, points=_points, wait=True)",
            "    _cli.close()",
        ]
        return [
            "from qdrant_client import QdrantClient",
            "from qdrant_client.http.models import VectorParams, Distance, PointStruct",
        ], lines


def _normalize_list(value, cls):
    if value is None:
        return []
    if isinstance(value, cls):
        return [value]
    return list(value)
