"""LlamaCppClientNode — in-process GGUF model handle via llama-cpp-python.

Unlike Ollama / OpenAI-compat (out-of-process HTTP), this backend loads the
model weights into the current Python process. Loading happens on the first
LLM method call, not at node execute, so wiring the node is cheap.
"""
from __future__ import annotations
from typing import Any

from core.node import BaseNode, PortType


class LlamaCppClientNode(BaseNode):
    type_name   = "ag_llama_cpp_client"
    label       = "llama.cpp Client"
    category    = "Agents"
    subcategory = "LLM"
    description = ("LLM client that loads a local GGUF model in-process via "
                   "llama-cpp-python. Slow to load; no daemon required.")

    def _setup_ports(self) -> None:
        self.add_input("model_path", PortType.STRING, default="",
                       description="Absolute path to a .gguf model file")
        self.add_input("n_ctx", PortType.INT, default=2048,
                       description="Context window tokens")
        self.add_input("n_gpu_layers", PortType.INT, default=0,
                       description="Layers to offload to GPU (0 = CPU-only; -1 = all)")
        self.add_input("chat_format", PortType.STRING, default="",
                       description=("Chat template override, e.g. "
                                    "'chatml-function-calling'. Empty = auto."))
        self.add_input("verbose", PortType.BOOL, default=False,
                       description="llama.cpp stderr chatter")
        self.add_output("llm", "LLM", description="Configured LLMClient")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._llm.llama_cpp_client import LlamaCppClient  # deferred
        path = (inputs.get("model_path") or "").strip()
        chat_format = (inputs.get("chat_format") or "").strip() or None
        return {"llm": LlamaCppClient(
            model_path=path,
            n_ctx=int(inputs.get("n_ctx") or 2048),
            n_gpu_layers=int(inputs.get("n_gpu_layers") or 0),
            chat_format=chat_format,
            verbose=bool(inputs.get("verbose")),
        )}

    def export(self, iv, ov):
        path = (self.inputs["model_path"].default_value or "").strip()
        n_ctx = int(self.inputs["n_ctx"].default_value or 2048)
        n_gpu = int(self.inputs["n_gpu_layers"].default_value or 0)
        chat_format = (self.inputs["chat_format"].default_value or "").strip()
        verbose = bool(self.inputs["verbose"].default_value)
        out = ov.get("llm", "_llamacpp")
        kwargs = [f"model_path={path!r}", f"n_ctx={n_ctx}", f"n_gpu_layers={n_gpu}"]
        if chat_format:
            kwargs.append(f"chat_format={chat_format!r}")
        if verbose:
            kwargs.append("verbose=True")
        lines = [
            f'{out} = {{"backend": "llama_cpp", '
            f'"client": Llama({", ".join(kwargs)}), "model": {path!r}}}',
        ]
        return ["from llama_cpp import Llama"], lines
