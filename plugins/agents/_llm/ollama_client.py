"""Ollama backend for LLMClient. Defers `import ollama` to first method call."""
from __future__ import annotations
import time
from typing import Any, ClassVar, Iterator

from plugins.agents._llm.protocol import (
    Message, ChatResult, ModelInfo,
)


class OllamaClient:
    """Talks to a local Ollama daemon over HTTP. Default host: http://localhost:11434.

    `import ollama` is deferred until first method call, so this class can be
    constructed (and the agents plugin can register) with the ollama package
    missing.
    """
    capabilities: ClassVar[set[str]] = {"tools", "embed", "stream", "json_schema"}

    def __init__(
        self,
        host: str = "http://localhost:11434",
        default_model: str | None = None,
    ):
        self.host = host
        self.default_model = default_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama  # deferred
            self._client = ollama.Client(host=self.host)
        return self._client

    def _resolve_model(self, model: str | None) -> str:
        m = model or self.default_model
        if not m:
            raise ValueError(
                "OllamaClient: no model specified (set default_model or pass model=)"
            )
        return m

    @staticmethod
    def _build_options(kw: dict) -> dict:
        """Translate normalized kwargs (temperature, max_tokens) into Ollama's options dict."""
        opts: dict[str, Any] = dict(kw.get("options") or {})
        if "temperature" in kw and kw["temperature"] is not None:
            opts["temperature"] = float(kw["temperature"])
        if "max_tokens" in kw and kw["max_tokens"] is not None:
            opts["num_predict"] = int(kw["max_tokens"])
        if "top_p" in kw and kw["top_p"] is not None:
            opts["top_p"] = float(kw["top_p"])
        return opts

    def chat(
        self, messages: list[Message], *, model: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None, **kw: Any,
    ) -> ChatResult:
        client = self._get_client()
        m = self._resolve_model(model)
        payload: dict[str, Any] = {
            "model": m,
            "messages": [msg.to_dict() for msg in messages],
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["format"] = response_format
        opts = self._build_options(kw)
        if opts:
            payload["options"] = opts

        t0 = time.time()
        resp = client.chat(**payload)
        elapsed_ms = (time.time() - t0) * 1000.0

        msg = resp.get("message", {}) or {}
        out_msg = Message(
            role=msg.get("role", "assistant"),
            content=msg.get("content", "") or "",
            tool_calls=msg.get("tool_calls"),
        )
        return ChatResult(
            message=out_msg, model=m,
            tokens_in=int(resp.get("prompt_eval_count", 0) or 0),
            tokens_out=int(resp.get("eval_count", 0) or 0),
            latency_ms=elapsed_ms, raw=resp,
        )

    def stream(
        self, messages: list[Message], *, model: str | None = None,
        tools: list[dict] | None = None, **kw: Any,
    ) -> Iterator[str]:
        client = self._get_client()
        m = self._resolve_model(model)
        payload: dict[str, Any] = {
            "model": m,
            "messages": [msg.to_dict() for msg in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        opts = self._build_options(kw)
        if opts:
            payload["options"] = opts
        for chunk in client.chat(**payload):
            piece = (chunk.get("message", {}) or {}).get("content", "")
            if piece:
                yield piece

    def embed(
        self, texts: list[str], *, model: str | None = None,
    ) -> list[list[float]]:
        client = self._get_client()
        m = model or "nomic-embed-text"
        # Ollama 0.3+ uses /api/embed (batch); fall back to /api/embeddings if needed.
        try:
            resp = client.embed(model=m, input=texts)
            return [list(v) for v in resp.get("embeddings", [])]
        except (AttributeError, TypeError):
            return [list(client.embeddings(model=m, prompt=t).get("embedding", []))
                    for t in texts]

    def list_models(self) -> list[ModelInfo]:
        client = self._get_client()
        resp = client.list()
        out: list[ModelInfo] = []
        for m in (resp.get("models", []) or []):
            # ollama 0.6+ Model objects expose the model name as "model" (not "name");
            # older versions used "name". Try both for forward + backward compat.
            name = m.get("model") or m.get("name") or ""
            out.append(ModelInfo(
                name=str(name),
                size_bytes=int(m.get("size", 0) or 0),
                modified=str(m.get("modified_at", "") or ""),
            ))
        return out

    def ping(self) -> bool:
        try:
            self.list_models()
            return True
        except Exception:
            return False
