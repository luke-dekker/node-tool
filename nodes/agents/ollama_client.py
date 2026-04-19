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
                       description="Model name (e.g., qwen2.5:7b). Leave blank to set later.")
        self.add_output("llm", "LLM", description="Configured LLMClient")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._llm.ollama_client import OllamaClient  # deferred
        host = inputs.get("host") or "http://localhost:11434"
        model = (inputs.get("model") or "").strip() or None
        return {"llm": OllamaClient(host=host, default_model=model)}

    def export(self, iv, ov):
        # Phase A export stub — full template-driven export lands in Phase D.
        return [], [f"# OllamaClientNode export pending Phase D"]
