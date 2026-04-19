"""LLMClient protocol + shared message / result dataclasses.

Backends (Ollama, OpenAI-compat, llama.cpp) implement this protocol. Each
adapter defers its heavy import until first method call, so importing this
module is free even if no backends are installed.

Convention for kwargs across backends: AgentNode passes normalized kwargs
(temperature=, max_tokens=) and each backend translates them to its native
format (Ollama: options dict; OpenAI: top-level kwargs).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Iterator, Protocol, runtime_checkable


@dataclass
class Message:
    role: str                              # system | user | assistant | tool
    content: str = ""
    name: str | None = None                # tool name when role == "tool"
    tool_calls: list[dict] | None = None   # populated on assistant messages
    tool_call_id: str | None = None        # references the call this tool msg responds to

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class ChatResult:
    message: Message
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    raw: Any = None        # backend-specific raw response


@dataclass
class ToolDef:
    """Tool definition bound to an AgentNode.

    `input_schema` is JSON Schema (the OpenAI / Ollama / llama.cpp common
    denominator). `callable` is invoked with kwargs parsed from the LLM's
    tool_call arguments. `side_effect` flags shell exec / file write / HTTP
    so AgentNode can gate them behind allow_side_effect_tools.
    """
    name: str
    description: str
    input_schema: dict = field(default_factory=lambda: {
        "type": "object", "properties": {}, "additionalProperties": True,
    })
    callable: Callable[..., Any] | None = None
    side_effect: bool = False

    def to_openai(self) -> dict:
        """Wire format for chat(tools=...) — works with Ollama and OpenAI-compat."""
        return {
            "type": "function",
            "function": {
                "name":        self.name,
                "description": self.description,
                "parameters":  self.input_schema,
            },
        }


@dataclass
class ModelInfo:
    name: str
    size_bytes: int = 0
    modified: str = ""

    @property
    def size_h(self) -> str:
        b = float(self.size_bytes)
        if b == 0:
            return "?"
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} PB"


@runtime_checkable
class LLMClient(Protocol):
    """Common surface every backend implements.

    `capabilities` declares which optional features the backend supports so
    node UIs can grey out unsupported toggles without try/except.
    """
    capabilities: ClassVar[set[str]]   # subset of {tools, embed, json_schema, stream}

    def chat(
        self, messages: list[Message], *, model: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None, **kw: Any,
    ) -> ChatResult: ...

    def stream(
        self, messages: list[Message], *, model: str | None = None,
        tools: list[dict] | None = None, **kw: Any,
    ) -> Iterator[str]: ...

    def embed(
        self, texts: list[str], *, model: str | None = None,
    ) -> list[list[float]]: ...

    def list_models(self) -> list[ModelInfo]: ...

    def ping(self) -> bool: ...
