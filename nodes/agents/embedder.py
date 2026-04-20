"""EmbedderNode — text → EMBEDDING[] via sentence-transformers.

Accepts either `documents: DOCUMENT[]` (preferred — passes through for
downstream store wiring) or `texts: STRING[]`. Returns aligned
`embeddings: EMBEDDING[]` plus pass-through `documents`.

Lazy-loads the `SentenceTransformer` model on first execute. The special
model name `hash-<dim>` selects the no-dep HashEmbedder for tests and
environments without torch.
"""
from __future__ import annotations
from typing import Any

from core.node import BaseNode, PortType


class EmbedderNode(BaseNode):
    type_name   = "ag_embedder"
    label       = "Embedder"
    category    = "Agents"
    subcategory = "Memory"
    description = ("Encode documents (or plain strings) into dense vectors. "
                   "Default: sentence-transformers all-MiniLM-L6-v2 (384-d).")

    def _setup_ports(self) -> None:
        self.add_input("documents", "DOCUMENT", default=None,
                       description="DOCUMENT[] to embed (preferred)")
        self.add_input("texts", PortType.STRING, default="",
                       description=("Fallback raw text (one per line) when "
                                    "no DOCUMENT[] is connected"))
        self.add_input("model", PortType.STRING, default="all-MiniLM-L6-v2",
                       description=("Sentence-transformers model name. Use "
                                    "'hash-<dim>' for the no-dep fallback."))
        self.add_input("device", PortType.STRING, default="",
                       description="cuda / cpu / mps; empty = auto")
        self.add_output("embeddings", "EMBEDDING")
        self.add_output("documents", "DOCUMENT")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._memory.embedder import build_embedder  # deferred
        from plugins.agents._memory.store_protocol import Document, Embedding  # deferred

        docs_in = inputs.get("documents")
        if docs_in is None or (isinstance(docs_in, list) and not docs_in):
            raw = inputs.get("texts") or ""
            lines = [ln for ln in str(raw).splitlines() if ln.strip()]
            docs = [Document(text=ln, metadata={"source": "<inline>", "chunk_idx": i})
                    for i, ln in enumerate(lines)]
        elif isinstance(docs_in, Document):
            docs = [docs_in]
        else:
            docs = list(docs_in)

        if not docs:
            return {"embeddings": [], "documents": []}

        model_name = (inputs.get("model") or "all-MiniLM-L6-v2").strip()
        device = (inputs.get("device") or "").strip() or None
        emb = build_embedder(model_name, device=device)

        vectors = emb.embed([d.text for d in docs])
        embeddings = [Embedding(vector=v, model=emb.model_name) for v in vectors]
        return {"embeddings": embeddings, "documents": docs}

    def export(self, iv, ov):
        docs_in = iv.get("documents") or "[]"
        texts_default = self.inputs["texts"].default_value or ""
        model = (self.inputs["model"].default_value or "all-MiniLM-L6-v2").strip()
        device = (self.inputs["device"].default_value or "").strip()

        embs_var = ov.get("embeddings", f"_embs_{self.safe_id}")
        docs_out = ov.get("documents",  f"_docs_out_{self.safe_id}")

        is_hash = model.startswith("hash")
        imports: list[str] = []
        lines: list[str] = [
            f"_docs_in = list({docs_in} or [])",
            "if not _docs_in:",
            f"    _lines = [ln for ln in {texts_default!r}.splitlines() if ln.strip()]",
            "    _docs_in = [{'text': _ln, 'metadata': {'source': '<inline>', "
            "'chunk_idx': _i}, 'id': f'<inline>:{_i}'} for _i, _ln in enumerate(_lines)]",
        ]
        if is_hash:
            dim = 256
            try:
                dim = int(model.split("-", 1)[1])
            except (IndexError, ValueError):
                pass
            lines.extend([
                f"_dim = {dim}",
                "def _ag_hash_embed(text, dim=_dim):",
                "    import math",
                "    v = [0.0] * dim",
                "    for tok in [t for t in str(text).lower().split() if t]:",
                "        idx = hash(('tok', tok)) % dim",
                "        sign = 1.0 if (hash(('sgn', tok)) & 1) else -1.0",
                "        v[idx] += sign",
                "    n = math.sqrt(sum(x*x for x in v)) or 1.0",
                "    return [x / n for x in v]",
                f"{embs_var} = [{{'vector': _ag_hash_embed(d['text']), 'dim': _dim, "
                f"'model': {model!r}}} for d in _docs_in]",
            ])
        else:
            imports.append("from sentence_transformers import SentenceTransformer")
            lines.extend([
                f"_emb_model = SentenceTransformer({model!r}"
                + (f", device={device!r}" if device else "") + ")",
                "_vecs = _emb_model.encode([d['text'] for d in _docs_in], "
                "convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)",
                f"{embs_var} = [{{'vector': list(map(float, v)), 'dim': len(v), "
                f"'model': {model!r}}} for v in _vecs]",
            ])
        lines.append(f"{docs_out} = _docs_in")
        return imports, lines
