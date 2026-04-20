"""RetrieverNode — rank DOCUMENTs from a MEMORY_REF by cosine similarity.

Takes a single query EMBEDDING + a MEMORY_REF. Returns top-k (document,
score) pairs, score-descending. The optional `where` dict is a flat
metadata filter (`{key: value}` or `{key: {"$in": [...]}}`) — the backend
translates it to its native DSL.
"""
from __future__ import annotations
import json
from typing import Any

from core.node import BaseNode, PortType


class RetrieverNode(BaseNode):
    type_name   = "ag_retriever"
    label       = "Retriever"
    category    = "Agents"
    subcategory = "Memory"
    description = ("Query a vector store by EMBEDDING, return top-k DOCUMENTs "
                   "with cosine scores.")

    def _setup_ports(self) -> None:
        self.add_input("query_embedding", "EMBEDDING", default=None,
                       description="Query vector (from EmbedderNode)")
        self.add_input("store_ref", "MEMORY_REF", default=None,
                       description="Handle from MemoryStoreNode")
        self.add_input("k", PortType.INT, default=5,
                       description="How many documents to return")
        self.add_input("where", PortType.STRING, default="",
                       description=("JSON metadata filter, e.g. "
                                    "{\"source\":\"docs.md\"}. Empty = no filter."))
        self.add_output("documents", "DOCUMENT")
        self.add_output("scores", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._memory.store_protocol import (  # deferred
            Embedding, MemoryRef, open_store,
        )

        q = inputs.get("query_embedding")
        ref = inputs.get("store_ref")
        if q is None:
            raise RuntimeError("RetrieverNode: query_embedding is required")
        if ref is None:
            raise RuntimeError("RetrieverNode: store_ref is required")
        if not isinstance(q, Embedding):
            raise RuntimeError(
                f"RetrieverNode: query_embedding must be an Embedding, got {type(q).__name__}"
            )
        if not isinstance(ref, MemoryRef):
            raise RuntimeError(
                f"RetrieverNode: store_ref must be a MemoryRef, got {type(ref).__name__}"
            )

        k = max(1, int(inputs.get("k") or 5))
        where_raw = (inputs.get("where") or "").strip()
        where = None
        if where_raw:
            try:
                where = json.loads(where_raw)
                if not isinstance(where, dict):
                    raise ValueError("where must decode to an object")
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"RetrieverNode: invalid where JSON: {exc}")

        store = open_store(ref)
        try:
            hits = store.query(q.vector, k=k, where=where)
        finally:
            store.close()

        docs = [d for d, _ in hits]
        scores = [s for _, s in hits]
        return {"documents": docs, "scores": scores}

    def export(self, iv, ov):
        q_var = iv.get("query_embedding") or "None  # TODO: wire a query embedding"
        ref_var = iv.get("store_ref") or "None  # TODO: wire a memory store_ref"
        k = int(self.inputs["k"].default_value or 5)
        where_raw = (self.inputs["where"].default_value or "").strip()
        docs_var   = ov.get("documents", f"_hits_docs_{self.safe_id}")
        scores_var = ov.get("scores",    f"_hits_scores_{self.safe_id}")

        lines: list[str] = [
            f"_q = {q_var}",
            f"_ref = {ref_var}",
            f"_k = {k}",
            f"_where_raw = {where_raw!r}",
        ]
        lines.extend([
            "import json as _json",
            "_where = _json.loads(_where_raw) if _where_raw else None",
            "_filter = None",
            "if _where:",
            "    _must = []",
            "    for _key, _val in _where.items():",
            "        _fkey = f'metadata.{_key}'",
            "        if isinstance(_val, dict) and '$in' in _val:",
            "            _must.append(FieldCondition(key=_fkey, "
            "match=MatchAny(any=list(_val['$in']))))",
            "        else:",
            "            _must.append(FieldCondition(key=_fkey, "
            "match=MatchValue(value=_val)))",
            "    _filter = Filter(must=_must)",
            "_cli = QdrantClient(path=_ref['path'])",
            "try:",
            "    if hasattr(_cli, 'query_points'):",
            "        _resp = _cli.query_points(collection_name=_ref['collection'],",
            "            query=list(_q['vector']), limit=_k, query_filter=_filter,",
            "            with_payload=True)",
            "        _hits = _resp.points",
            "    else:",
            "        _hits = _cli.search(collection_name=_ref['collection'],",
            "            query_vector=list(_q['vector']), limit=_k, "
            "query_filter=_filter)",
            f"    {docs_var} = [{{'id': str(h.id), 'text': (h.payload or {{}}).get('text',''), "
            "'metadata': (h.payload or {}).get('metadata', {})} for h in _hits]",
            f"    {scores_var} = [float(h.score) for h in _hits]",
            "finally:",
            "    _cli.close()",
        ])
        return [
            "from qdrant_client import QdrantClient",
            "from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny",
        ], lines
