"""Ollama and Agno inference nodes."""

from __future__ import annotations
from typing import Any
from core.node import BaseNode
from core.node import PortType

CATEGORY = "AI"


# ── Ollama Generate ────────────────────────────────────────────────────────────

class OllamaGenerateNode(BaseNode):
    type_name   = "ai_ollama_generate"
    label       = "Ollama Generate"
    category    = CATEGORY
    subcategory = "Ollama"
    description = (
        "Call a local Ollama model (localhost:11434). "
        "Fast direct inference — no agent overhead."
    )

    def _setup_ports(self) -> None:
        self.add_input("prompt",      PortType.STRING, default="Hello!",
                       description="User message / prompt")
        self.add_input("model",       PortType.STRING, default="qwen3:32b-q4_K_M",
                       description="Ollama model name (ollama list to see available)")
        self.add_input("system",      PortType.STRING, default="",
                       description="System prompt (leave blank for none)")
        self.add_input("temperature", PortType.FLOAT,  default=0.7)
        self.add_input("max_tokens",  PortType.INT,    default=512,
                       description="Maximum tokens to generate")
        self.add_input("host",        PortType.STRING, default="http://localhost:11434",
                       description="Ollama host URL")
        self.add_output("response",    PortType.STRING)
        self.add_output("__terminal__", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        prompt      = inputs.get("prompt") or ""
        model       = inputs.get("model")  or "qwen3:32b-q4_K_M"
        system      = inputs.get("system") or ""
        temperature = float(inputs.get("temperature") or 0.7)
        max_tokens  = int(inputs.get("max_tokens") or 512)
        host        = (inputs.get("host") or "http://localhost:11434").rstrip("/")

        if not prompt:
            return {"response": "", "__terminal__": "[Ollama] No prompt."}

        try:
            import requests, json

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model":   model,
                "messages": messages,
                "stream":  False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            resp = requests.post(f"{host}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            data     = resp.json()
            response = data.get("message", {}).get("content", "")
            log = f"[Ollama] {model} → {len(response)} chars"
            return {"response": response, "__terminal__": log}

        except Exception as exc:
            msg = f"[Ollama] Error: {exc}"
            return {"response": "", "__terminal__": msg}


# ── Ollama Embed ───────────────────────────────────────────────────────────────

class OllamaEmbedNode(BaseNode):
    type_name   = "ai_ollama_embed"
    label       = "Ollama Embed"
    category    = CATEGORY
    subcategory = "Ollama"
    description = (
        "Get an embedding vector from a local Ollama model. "
        "Returns a 1-D numpy array you can wire into NumPy / sklearn nodes."
    )

    def _setup_ports(self) -> None:
        self.add_input("text",   PortType.STRING, default="",
                       description="Text to embed")
        self.add_input("model",  PortType.STRING, default="nomic-embed-text",
                       description="Embedding model name")
        self.add_input("host",   PortType.STRING, default="http://localhost:11434")
        self.add_output("embedding",   PortType.NDARRAY,
                        description="1-D float32 numpy array")
        self.add_output("__terminal__", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text  = inputs.get("text")  or ""
        model = inputs.get("model") or "nomic-embed-text"
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


# ── Agno Agent ─────────────────────────────────────────────────────────────────

class AgnoAgentNode(BaseNode):
    type_name   = "ai_agno_agent"
    label       = "Agno Agent"
    category    = CATEGORY
    subcategory = "Agno"
    description = (
        "Call an Agno agent or team (localhost:8000). "
        "Use 'coding-team' to route through the full multi-agent team "
        "(leader + coder + researcher with Qdrant knowledge base)."
    )

    def _setup_ports(self) -> None:
        self.add_input("message",  PortType.STRING, default="",
                       description="Message to send to the agent")
        self.add_input("agent_id", PortType.STRING, default="coding-team",
                       description="Agent or team id: coding-team | coder | researcher")
        self.add_input("host",     PortType.STRING, default="http://localhost:8000",
                       description="Agno FastAPI host")
        self.add_input("stream",   PortType.BOOL,   default=False,
                       description="Use SSE streaming (returns concatenated response)")
        self.add_output("response",    PortType.STRING)
        self.add_output("__terminal__", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        message  = inputs.get("message")  or ""
        agent_id = inputs.get("agent_id") or "coding-team"
        host     = (inputs.get("host") or "http://localhost:8000").rstrip("/")
        use_stream = bool(inputs.get("stream", False))

        if not message:
            return {"response": "", "__terminal__": "[Agno] No message."}

        try:
            import requests

            # Determine endpoint: teams vs agents
            if "team" in agent_id:
                url = f"{host}/v1/teams/{agent_id}/runs"
            else:
                url = f"{host}/v1/agents/{agent_id}/runs"

            payload = {"message": message}

            if use_stream:
                # Collect SSE stream
                lines = []
                with requests.post(url, json=payload, stream=True, timeout=180) as r:
                    r.raise_for_status()
                    for raw in r.iter_lines():
                        if not raw:
                            continue
                        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                        if line.startswith("data:"):
                            chunk = line[5:].strip()
                            if chunk and chunk != "[DONE]":
                                try:
                                    import json
                                    obj = json.loads(chunk)
                                    content = (
                                        obj.get("content")
                                        or obj.get("message", {}).get("content")
                                        or ""
                                    )
                                    lines.append(content)
                                except Exception:
                                    lines.append(chunk)
                response = "".join(lines)
            else:
                resp = requests.post(url, json=payload, timeout=180)
                resp.raise_for_status()
                data = resp.json()
                response = (
                    data.get("content")
                    or data.get("message", {}).get("content")
                    or str(data)
                )

            log = f"[Agno] {agent_id} → {len(response)} chars"
            return {"response": response, "__terminal__": log}

        except Exception as exc:
            msg = f"[Agno] Error: {exc}"
            return {"response": "", "__terminal__": msg}
