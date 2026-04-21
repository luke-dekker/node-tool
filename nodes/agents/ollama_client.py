"""OllamaClientNode — emits an LLM port handle to a configured Ollama backend."""
from __future__ import annotations
from typing import Any

from core.node import BaseNode, PortType


class OllamaClientNode(BaseNode):
    type_name   = "ag_ollama_client"
    label       = "Ollama Client"
    category    = "Agents"
    subcategory = "LLM"
    description = "LLM client backed by a local Ollama daemon."

    def _setup_ports(self) -> None:
        self.add_input("host", PortType.STRING, default="http://localhost:11434",
                       description="Ollama daemon URL")
        self.add_input("model", PortType.STRING, default="",
                       description="Model name (e.g., qwen2.5:7b). Inspector "
                                   "offers a dropdown of installed models.",
                       dynamic_choices="agent_list_local_models")
        self.add_output("llm", "LLM", description="Configured LLMClient")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._llm.ollama_client import OllamaClient  # deferred
        host = inputs.get("host") or "http://localhost:11434"
        model = (inputs.get("model") or "").strip() or None
        return {"llm": OllamaClient(host=host, default_model=model)}

    def export(self, iv, ov):
        host = (self.inputs["host"].default_value or "http://localhost:11434").strip()
        model = (self.inputs["model"].default_value or "").strip()
        out = ov.get("llm", "_ollama")
        lines = [
            f'{out} = {{"backend": "ollama", '
            f'"client": Client(host={host!r}), '
            f'"model": {model!r}}}'
        ]
        return ["from ollama import Client"], lines
