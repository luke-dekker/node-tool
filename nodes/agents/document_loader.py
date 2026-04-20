"""DocumentLoaderNode — read a text file and split into chunks.

Output `documents: DOCUMENT[]` with `{source, chunk_idx}` metadata attached,
plus a stable content-addressed `id` per chunk so repeated runs upsert
idempotently against the vector store.

v1 supports .txt / .md natively and falls back to reading any file as UTF-8
text (with replacement on decode errors). PDF / HTML can be added in a
later pass; the splitter is format-agnostic.
"""
from __future__ import annotations
import os
from typing import Any

from core.node import BaseNode, PortType


class DocumentLoaderNode(BaseNode):
    type_name   = "ag_document_loader"
    label       = "Document Loader"
    category    = "Agents"
    subcategory = "Memory"
    description = ("Read a text file, split into overlapping chunks, emit "
                   "DOCUMENT[] with {source, chunk_idx} metadata.")

    def _setup_ports(self) -> None:
        self.add_input("path", PortType.STRING, default="",
                       description="File path (txt/md; UTF-8 read for others)")
        self.add_input("text", PortType.STRING, default="",
                       description="Raw text to chunk (used if path is empty)")
        self.add_input("chunk_size", PortType.INT, default=512,
                       description="Target chunk length in characters")
        self.add_input("chunk_overlap", PortType.INT, default=64,
                       description="Characters of overlap between chunks")
        self.add_output("documents", "DOCUMENT")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._memory.store_protocol import Document  # deferred

        path = (inputs.get("path") or "").strip()
        raw = inputs.get("text") or ""
        source = path or "<inline>"

        if path:
            if not os.path.isfile(path):
                raise RuntimeError(f"DocumentLoaderNode: file not found: {path}")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()

        if not raw.strip():
            return {"documents": []}

        chunk_size = max(1, int(inputs.get("chunk_size") or 512))
        overlap = max(0, min(chunk_size - 1, int(inputs.get("chunk_overlap") or 0)))

        chunks = _chunk_text(raw, chunk_size, overlap)
        docs = [
            Document(
                text=chunk,
                metadata={"source": source, "chunk_idx": idx},
            )
            for idx, chunk in enumerate(chunks)
        ]
        return {"documents": docs}

    def export(self, iv, ov):
        path = (self.inputs["path"].default_value or "").strip()
        text_default = self.inputs["text"].default_value or ""
        chunk_size = int(self.inputs["chunk_size"].default_value or 512)
        overlap = int(self.inputs["chunk_overlap"].default_value or 64)

        docs_var = ov.get("documents", f"_docs_{self.safe_id}")
        lines: list[str] = [
            f"_path = {path!r}",
            f"_raw = {text_default!r}",
            "if _path:",
            "    with open(_path, 'r', encoding='utf-8', errors='replace') as _f:",
            "        _raw = _f.read()",
            f"_chunk = {chunk_size}",
            f"_ovlp = {overlap}",
            "_step = max(1, _chunk - _ovlp)",
            "_chunks = []",
            "if _raw.strip():",
            "    _i = 0",
            "    while _i < len(_raw):",
            "        _chunks.append(_raw[_i:_i + _chunk])",
            "        if _i + _chunk >= len(_raw): break",
            "        _i += _step",
            "_source = _path or '<inline>'",
            f"{docs_var} = [{{'text': _c, 'metadata': {{'source': _source, "
            f"'chunk_idx': _i}}, 'id': f'{{_source}}:{{_i}}'}} for _i, _c in enumerate(_chunks)]",
        ]
        return [], lines


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size sliding window over characters. Last chunk may be shorter.

    Character-based (not token-based) for zero-dependency chunking. Good
    enough for RAG over markdown / source; for token-accurate chunking,
    wire in `tiktoken` via a later EmbedderNode option.
    """
    if not text:
        return []
    step = max(1, size - overlap)
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        out.append(text[i:i + size])
        if i + size >= n:
            break
        i += step
    return out
