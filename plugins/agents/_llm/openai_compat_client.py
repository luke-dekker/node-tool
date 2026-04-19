"""OpenAI-compatible backend.

Works against LM Studio, llama.cpp's HTTP server, vLLM, Ollama's /v1/* surface,
OpenRouter, etc. Defers `import openai` to first method call.
"""
from __future__ import annotations
import time
from typing import Any, ClassVar, Iterator

from plugins.agents._llm.protocol import (
    Message, ChatResult, ModelInfo,
)


class OpenAICompatClient:
    capabilities: ClassVar[set[str]] = {"tools", "embed", "stream", "json_schema"}

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        default_model: str | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.default_model = default_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai  # deferred
            self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _resolve_model(self, model: str | None) -> str:
        m = model or self.default_model
        if not m:
            raise ValueError("OpenAICompatClient: no model specified")
        return m

    def chat(
        self, messages: list[Message], *, model: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None, **kw: Any,
    ) -> ChatResult:
        client = self._get_client()
        m = self._resolve_model(model)
        kwargs: dict[str, Any] = {
            "model": m,
            "messages": [msg.to_dict() for msg in messages],
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        for k in ("temperature", "max_tokens", "top_p"):
            if k in kw and kw[k] is not None:
                kwargs[k] = kw[k]

        t0 = time.time()
        resp = client.chat.completions.create(**kwargs)
        elapsed_ms = (time.time() - t0) * 1000.0

        choice = resp.choices[0].message
        out_msg = Message(
            role=choice.role,
            content=choice.content or "",
            tool_calls=[tc.model_dump() for tc in (choice.tool_calls or [])] or None,
        )
        usage = getattr(resp, "usage", None)
        return ChatResult(
            message=out_msg, model=m,
            tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=elapsed_ms, raw=resp,
        )

    def stream(
        self, messages: list[Message], *, model: str | None = None,
        tools: list[dict] | None = None, **kw: Any,
    ) -> Iterator[str]:
        client = self._get_client()
        m = self._resolve_model(model)
        kwargs: dict[str, Any] = {
            "model": m,
            "messages": [msg.to_dict() for msg in messages],
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
        for k in ("temperature", "max_tokens", "top_p"):
            if k in kw and kw[k] is not None:
                kwargs[k] = kw[k]
        for chunk in client.chat.completions.create(**kwargs):
            delta = chunk.choices[0].delta
            piece = (delta.content or "") if delta else ""
            if piece:
                yield piece

    def embed(
        self, texts: list[str], *, model: str | None = None,
    ) -> list[list[float]]:
        client = self._get_client()
        m = model or "text-embedding-3-small"
        resp = client.embeddings.create(model=m, input=texts)
        return [list(d.embedding) for d in resp.data]

    def list_models(self) -> list[ModelInfo]:
        client = self._get_client()
        resp = client.models.list()
        return [ModelInfo(name=m.id) for m in resp.data]

    def ping(self) -> bool:
        try:
            self.list_models()
            return True
        except Exception:
            return False
