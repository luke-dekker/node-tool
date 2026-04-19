"""MockLLMClient — deterministic stub for tests. No network calls."""
from __future__ import annotations
from typing import Any, ClassVar, Iterator

from plugins.agents._llm.protocol import (
    Message, ChatResult, ModelInfo,
)


class MockLLMClient:
    """Records every call; emits a canned response."""
    capabilities: ClassVar[set[str]] = {"tools", "embed", "stream"}

    def __init__(
        self,
        response: str = "ok",
        tool_calls: list[dict] | None = None,
        models: list[str] | None = None,
    ):
        self.response = response
        self.tool_calls = tool_calls
        self._models = models or ["mock-model"]
        self.calls: list[dict] = []

    def chat(self, messages, *, model=None, tools=None, response_format=None, **kw):
        self.calls.append({"messages": list(messages), "tools": tools, "kw": dict(kw)})
        out = Message(
            role="assistant",
            content=self.response,
            tool_calls=self.tool_calls,
        )
        return ChatResult(
            message=out, model=model or "mock",
            tokens_in=10, tokens_out=5, latency_ms=1.0, raw=None,
        )

    def stream(self, messages, *, model=None, tools=None, **kw) -> Iterator[str]:
        self.calls.append({"messages": list(messages), "tools": tools, "kw": dict(kw),
                           "stream": True})
        for ch in self.response:
            yield ch

    def embed(self, texts, *, model=None):
        # Dummy 3-d embeddings; deterministic per text length.
        return [[float(len(t)), 0.0, 0.0] for t in texts]

    def list_models(self):
        return [ModelInfo(name=n, size_bytes=1000) for n in self._models]

    def ping(self) -> bool:
        return True
