"""Ollama Generate node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class OllamaGenerateNode(BaseNode):
    type_name   = "ai_ollama_generate"
    label       = "Ollama Generate"
    category    = "Models"
    subcategory = "AI"
    description = (
        "Call a local Ollama model. "
        "Requires Ollama running (ollama.com)."
    )

    def _setup_ports(self) -> None:
        self.add_input("prompt",      PortType.STRING, default="Hello!",
                       description="User message / prompt")
        self.add_input("model",       PortType.STRING, default="llama3.1:8b",
                       description="Ollama model name (run 'ollama list' to see available)")
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
        model       = inputs.get("model")  or "llama3.1:8b"
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

    def export(self, iv, ov):
        prompt = self._val(iv, "prompt")
        model  = self._val(iv, "model")
        system = self._val(iv, "system")
        temp   = self._val(iv, "temperature")
        maxt   = self._val(iv, "max_tokens")
        host   = self._val(iv, "host")
        out    = ov.get("response", "_ollama_response")
        lines  = [
            "import requests as _req",
            f"_ollama_msgs = []",
            f"if {system}: _ollama_msgs.append({{'role': 'system', 'content': {system}}})",
            f"_ollama_msgs.append({{'role': 'user', 'content': {prompt}}})",
            f"_ollama_r = _req.post({host} + '/api/chat', json={{'model': {model}, 'messages': _ollama_msgs, 'stream': False, 'options': {{'temperature': {temp}, 'num_predict': {maxt}}}}}, timeout=120)",
            f"{out} = _ollama_r.json().get('message', {{}}).get('content', '')",
        ]
        return [], lines
