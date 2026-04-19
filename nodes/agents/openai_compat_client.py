"""OpenAICompatClientNode — LLM handle for any OpenAI-compatible endpoint."""
from __future__ import annotations
from typing import Any

from core.node import BaseNode, PortType


class OpenAICompatClientNode(BaseNode):
    type_name   = "ag_openai_compat_client"
    label       = "OpenAI-Compat Client"
    category    = "Agents"
    subcategory = "LLM"
    description = "LLM client for any OpenAI-compatible endpoint (LM Studio, vLLM, OpenRouter, ...)."

    def _setup_ports(self) -> None:
        self.add_input("base_url", PortType.STRING,
                       default="http://localhost:11434/v1",
                       description="Base URL of the OpenAI-compat endpoint")
        self.add_input("api_key", PortType.STRING, default="ollama",
                       description="API key (any non-empty string for local backends)")
        self.add_input("model", PortType.STRING, default="",
                       description="Model name (server-specific)")
        self.add_output("llm", "LLM")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._llm.openai_compat_client import OpenAICompatClient  # deferred
        base_url = inputs.get("base_url") or "http://localhost:11434/v1"
        api_key  = inputs.get("api_key")  or "ollama"
        model    = (inputs.get("model") or "").strip() or None
        return {"llm": OpenAICompatClient(base_url=base_url, api_key=api_key,
                                          default_model=model)}

    def export(self, iv, ov):
        return [], [f"# OpenAICompatClientNode export pending Phase D"]
