"""Ollama Embed node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class OllamaEmbedNode(BaseNode):
    type_name   = "ai_ollama_embed"
    label       = "Ollama Embed"
    category    = "AI"
    subcategory = "Ollama"
    description = (
        "Get an embedding vector from a local Ollama model. "
        "Returns a 1-D numpy array. Requires Ollama running (ollama.com)."
    )

    def _setup_ports(self) -> None:
        self.add_input("text",   PortType.STRING, default="",
                       description="Text to embed")
        self.add_input("model",  PortType.STRING, default="all-minilm",
                       description="Embedding model name")
        self.add_input("host",   PortType.STRING, default="http://localhost:11434")
        self.add_output("embedding",   PortType.NDARRAY,
                        description="1-D float32 numpy array")
        self.add_output("__terminal__", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text  = inputs.get("text")  or ""
        model = inputs.get("model") or "all-minilm"
        host  = (inputs.get("host") or "http://localhost:11434").rstrip("/")

        if not text:
            return {"embedding": None, "__terminal__": "[Embed] No text."}

        try:
            import requests
            import numpy as np

            payload = {"model": model, "input": text}
            resp = requests.post(f"{host}/api/embed", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # Ollama /api/embed returns {"embeddings": [[...]]}
            vec  = data.get("embeddings", [[]])[0]
            arr  = np.array(vec, dtype=np.float32)
            log  = f"[Embed] {model} → {arr.shape[0]}-dim vector"
            return {"embedding": arr, "__terminal__": log}

        except Exception as exc:
            return {"embedding": None, "__terminal__": f"[Embed] Error: {exc}"}

    def export(self, iv, ov):
        text  = self._val(iv, "text")
        model = self._val(iv, "model")
        host  = self._val(iv, "host")
        out   = ov.get("embedding", "_ollama_embed")
        lines = [
            "import requests as _req, numpy as _np",
            f"_embed_r = _req.post({host} + '/api/embed', json={{'model': {model}, 'input': {text}}}, timeout=60)",
            f"{out} = _np.array(_embed_r.json().get('embeddings', [[]])[0], dtype=_np.float32)",
        ]
        return [], lines
