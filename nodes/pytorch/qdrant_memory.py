"""QdrantMemoryNode — vector-DB memory bank for tensor inputs.

The "embedder" is whatever upstream layer's output you wire in — typically
a frozen pretrained encoder. The node:

  • indexes those vectors into a persistent Qdrant collection
  • queries top-K nearest neighbours given a query vector
  • returns retrieved vectors as a (B, K, D) tensor you can concatenate
    or attend over downstream

Three modes via the `mode` dropdown:

  query             — read-only; collection populated by a prior pass
  index             — write-only; one-shot setup or per-step bookkeeping
  query_then_index  — query first (retrieved excludes the current step's
                      write), then index. The standard "online memory
                      bank" pattern for memory-augmented LMs.

Retrieved tensors are detached — gradients don't flow back into the
indexed vectors, so use this with a frozen embedder. The downstream
model learns to *use* what comes back.

Empty collection on first query returns zeros (B, K, D). Top-K larger
than the collection pads with zeros.

Persistent storage: `./qdrant_data` by default (local mode, no server).
Pass a URL like `http://localhost:6333` to point at a remote server.
"""
from __future__ import annotations
import uuid
from typing import Any
from core.node import BaseNode, InspectorSpec, PortType


_MODES = ["query", "index", "query_then_index"]

# Qdrant local mode acquires an exclusive lock on its data directory, so
# only one client per `path` can exist in a process. Cache by path so
# multiple QdrantMemoryNode instances pointing at the same store share
# a client instead of fighting over the lock.
_CLIENT_CACHE: dict[str, "object"] = {}


class QdrantMemoryNode(BaseNode):
    type_name   = "pt_qdrant_memory"
    label       = "Qdrant Memory"
    category    = "Memory"
    subcategory = ""
    description = (
        "Vector-DB memory bank backed by Qdrant. Wire any (B, D) tensor "
        "from a frozen embedder into `query_embedding`; in `query` or "
        "`query_then_index` mode the node returns the top-K nearest "
        "neighbours as a (B, K, D) tensor.\n"
        "  • collection persists at `path` (default ./qdrant_data) — your\n"
        "    memory bank survives across runs\n"
        "  • use the inspector's 'Clear collection' button to wipe it\n"
        "  • retrieved tensor is detached: use a frozen embedder."
    )

    def __init__(self):
        self._collection_dim: int | None = None
        self._last_count = 0
        self._last_info = ""
        super().__init__()

    def relevant_inputs(self, values):
        mode = (values.get("mode") or "query").strip()
        base = ["mode", "collection", "path"]
        if mode in ("query", "query_then_index"):
            base.append("top_k")
        return base

    def _setup_ports(self):
        self.add_input("query_embedding", PortType.TENSOR, default=None,
                       description="(B, D) query vectors. Last dim = embedding dim.")
        self.add_input("write_embedding", PortType.TENSOR, default=None, optional=True,
                       description="(B, D) vectors to upsert. Defaults to query_embedding "
                                   "when mode includes write. Last dim must match.")
        self.add_input("mode",            PortType.STRING, default="query", choices=_MODES)
        self.add_input("collection",      PortType.STRING, default="rnn_memory")
        self.add_input("path",            PortType.STRING, default="./qdrant_data",
                       description="Local persistent dir, or http(s):// URL for remote.")
        self.add_input("top_k",           PortType.INT, default=4)
        self.add_output("retrieved",      PortType.TENSOR,
                        description="(B, K, D) detached top-K retrieved vectors. Zeros "
                                    "where the collection has fewer than K points.")
        self.add_output("count",          PortType.INT,
                        description="Collection size after the op.")
        self.add_output("info",           PortType.STRING)

    # ── Client management ──────────────────────────────────────────────────

    def _get_client(self, path: str):
        cached = _CLIENT_CACHE.get(path)
        if cached is not None:
            return cached
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise RuntimeError("qdrant-client not installed — pip install qdrant-client")
        if path.startswith(("http://", "https://")):
            client = QdrantClient(url=path)
        else:
            client = QdrantClient(path=path)
        _CLIENT_CACHE[path] = client
        return client

    def _ensure_collection(self, client, collection: str, dim: int) -> None:
        from qdrant_client.http import models as qm
        existing = {c.name for c in client.get_collections().collections}
        if collection in existing:
            info = client.get_collection(collection)
            existing_dim = info.config.params.vectors.size
            if existing_dim != dim:
                raise RuntimeError(
                    f"collection {collection!r} has dim {existing_dim}, "
                    f"got vectors with dim {dim} — clear or use a different collection"
                )
            self._collection_dim = existing_dim
            return
        client.create_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        self._collection_dim = dim

    # ── Execute ────────────────────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            import torch
            q = inputs.get("query_embedding")
            mode = (inputs.get("mode") or "query").strip()
            collection = (inputs.get("collection") or "rnn_memory").strip()
            path = (inputs.get("path") or "./qdrant_data").strip()
            top_k = max(1, int(inputs.get("top_k") or 4))

            if q is None or not isinstance(q, torch.Tensor):
                self._last_info = "no query_embedding wired"
                return {"retrieved": None, "count": self._last_count, "info": self._last_info}
            if q.dim() != 2:
                self._last_info = f"query_embedding must be (B, D); got shape {tuple(q.shape)}"
                return {"retrieved": None, "count": self._last_count, "info": self._last_info}

            B, D = q.shape
            client = self._get_client(path)
            self._ensure_collection(client, collection, D)

            retrieved = None
            # Query first, BEFORE writing — so query_then_index doesn't see
            # the current step's own writes (the standard online-memory pattern).
            if mode in ("query", "query_then_index"):
                retrieved = self._query(
                    client, collection, q, top_k, dtype=q.dtype, device=q.device,
                )

            if mode in ("index", "query_then_index"):
                w = inputs.get("write_embedding")
                if w is None:
                    w = q
                if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.shape[-1] != D:
                    self._last_info = (f"write_embedding must be (B, {D}); "
                                       f"got {type(w).__name__} shape "
                                       f"{tuple(w.shape) if isinstance(w, torch.Tensor) else '?'}")
                    return {"retrieved": retrieved, "count": self._last_count,
                            "info": self._last_info}
                self._index(client, collection, w)

            count = int(client.count(collection_name=collection, exact=True).count)
            self._last_count = count
            self._last_info = (f"mode={mode} collection={collection!r} count={count} "
                               f"dim={D} k={top_k if retrieved is not None else '-'}")
            return {"retrieved": retrieved, "count": count, "info": self._last_info}
        except Exception as exc:
            self._last_info = f"qdrant memory failed: {exc}"
            return {"retrieved": None, "count": self._last_count, "info": self._last_info}

    def _query(self, client, collection: str, q, top_k: int, dtype, device):
        import torch
        B, D = q.shape
        out = torch.zeros(B, top_k, D, dtype=dtype, device=device)
        # Detach + cpu for the API call; retrieved tensor is constructed
        # fresh so it has no grad lineage either way.
        q_cpu = q.detach().cpu()
        for i in range(B):
            vec = q_cpu[i].tolist()
            try:
                if hasattr(client, "query_points"):
                    resp = client.query_points(
                        collection_name=collection, query=vec,
                        limit=top_k, with_payload=True, with_vectors=True,
                    )
                    hits = resp.points
                else:
                    hits = client.search(
                        collection_name=collection, query_vector=vec,
                        limit=top_k, with_payload=True, with_vectors=True,
                    )
            except Exception:
                continue
            for k, h in enumerate(hits[:top_k]):
                # Prefer raw vector from payload (unit-magnitude preserved);
                # fall back to h.vector (Qdrant pre-normalises for cosine).
                raw = (h.payload or {}).get("raw") if h.payload else None
                src = raw if raw is not None else h.vector
                if src is None:
                    continue
                out[i, k] = torch.as_tensor(src, dtype=dtype, device=device)
        return out

    def _index(self, client, collection: str, w) -> None:
        from qdrant_client.http import models as qm
        w_cpu = w.detach().cpu()
        points = [
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=row.tolist(),
                payload={"raw": row.tolist()},   # original magnitude preserved
            )
            for row in w_cpu
        ]
        if points:
            client.upsert(collection_name=collection, points=points, wait=True)

    # ── Inspector: clear-collection button ────────────────────────────────

    def inspector_spec(self):
        path = (self.inputs["path"].default_value or "./qdrant_data")
        coll = (self.inputs["collection"].default_value or "rnn_memory")
        lines = [
            f"path:       {path}",
            f"collection: {coll}",
            f"count:      {self._last_count}",
            f"last:       {self._last_info or '(not run)'}",
        ]
        return InspectorSpec(
            section="Qdrant Memory",
            lines=lines,
            actions=[("Clear collection", "clear_collection")],
        )

    def clear_collection(self, app=None) -> dict:
        """Delete every point in the collection. Schema/dim is preserved.

        Uses delete-by-filter (match-all) instead of delete_collection +
        recreate, because Qdrant local mode keeps the on-disk storage
        for the collection name and pulls stale points back in on the
        next create. Delete-by-filter is the reliable wipe.
        """
        try:
            from qdrant_client.http import models as qm
            path = (self.inputs["path"].default_value or "./qdrant_data")
            collection = (self.inputs["collection"].default_value or "rnn_memory")
            client = self._get_client(path)
            existing = {c.name for c in client.get_collections().collections}
            if collection not in existing:
                self._last_info = f"collection {collection!r} did not exist"
                return {"text": self._last_info}
            client.delete(
                collection_name=collection,
                points_selector=qm.FilterSelector(filter=qm.Filter(must=[])),
                wait=True,
            )
            self._last_count = 0
            self._last_info = f"cleared {collection!r}"
            return {"text": f"cleared collection {collection!r}"}
        except Exception as exc:
            return {"error": f"clear failed: {exc}"}

    # ── Export ─────────────────────────────────────────────────────────────

    def export(self, iv, ov):
        q = iv.get("query_embedding") or "None  # TODO: wire query_embedding"
        w = iv.get("write_embedding") or q
        mode = (self.inputs["mode"].default_value or "query")
        coll = (self.inputs["collection"].default_value or "rnn_memory")
        path = (self.inputs["path"].default_value or "./qdrant_data")
        top_k = int(self.inputs["top_k"].default_value or 4)
        ret_v = ov.get("retrieved", "_retrieved")
        cnt_v = ov.get("count",     "_count")
        inf_v = ov.get("info",      "_info")
        return [
            "import torch",
            "import uuid",
            "from qdrant_client import QdrantClient",
            "from qdrant_client.http import models as _qm",
        ], [
            f"_q = {q}",
            f"_path = {path!r}",
            f"_coll = {coll!r}",
            f"_top_k = {top_k}",
            f"_cli = QdrantClient(url=_path) if _path.startswith(('http://','https://')) "
            f"else QdrantClient(path=_path)",
            f"_dim = _q.shape[-1]",
            f"_existing = {{c.name for c in _cli.get_collections().collections}}",
            f"if _coll not in _existing:",
            f"    _cli.create_collection(collection_name=_coll, "
            f"vectors_config=_qm.VectorParams(size=_dim, distance=_qm.Distance.COSINE))",
            f"_B = _q.shape[0]",
            f"{ret_v} = torch.zeros(_B, _top_k, _dim, dtype=_q.dtype, device=_q.device)",
            f"if {mode!r} in ('query', 'query_then_index'):",
            f"    for _i in range(_B):",
            f"        _resp = _cli.query_points(collection_name=_coll, "
            f"query=_q[_i].detach().cpu().tolist(), limit=_top_k, with_vectors=True)",
            f"        for _k, _h in enumerate(_resp.points[:_top_k]):",
            f"            if _h.vector is not None:",
            f"                {ret_v}[_i, _k] = torch.as_tensor(_h.vector, "
            f"dtype=_q.dtype, device=_q.device)",
            f"if {mode!r} in ('index', 'query_then_index'):",
            f"    _w = {w}",
            f"    _pts = []",
            f"    for _r in _w:",
            f"        _v = _r.detach().cpu().tolist()",
            f"        _pts.append(_qm.PointStruct(id=str(uuid.uuid4()), "
            f"vector=_v, payload={{'raw': _v}}))",
            f"    _cli.upsert(collection_name=_coll, points=_pts, wait=True)",
            f"{cnt_v} = int(_cli.count(collection_name=_coll, exact=True).count)",
            f"{inf_v} = f'mode={mode} count={{{cnt_v}}}'",
        ]
