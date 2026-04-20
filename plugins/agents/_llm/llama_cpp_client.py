"""llama-cpp-python in-process backend for LLMClient.

Unlike Ollama / OpenAI-compat (out-of-process HTTP), llama-cpp is in-process:
one `Llama(model_path=…)` instance owns the model weights and (optionally) a
chunk of GPU memory. Loading is expensive, so construction of the handle is
cheap and the `Llama` object is built on first method call.

Tool calling works through llama-cpp's OpenAI-shaped `create_chat_completion`;
pass `chat_format="chatml-function-calling"` for models that need it, or leave
unset for newer GGUFs with a native tool-call chat template baked in.

Embeddings require a separately-constructed instance with `embedding=True`,
so `embed()` lazy-builds a second `Llama` handle if the primary one wasn't
created for embedding.
"""
from __future__ import annotations
import os
import time
from typing import Any, ClassVar, Iterator

from plugins.agents._llm.protocol import (
    Message, ChatResult, ModelInfo,
)


class LlamaCppClient:
    """Talks to a local GGUF model via `llama-cpp-python`.

    `import llama_cpp` is deferred until first method call, so this class can
    be constructed (and the agents plugin can register) with the llama_cpp
    package missing.
    """
    capabilities: ClassVar[set[str]] = {"tools", "stream", "embed", "json_schema"}

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        chat_format: str | None = None,
        verbose: bool = False,
        **extra_kw: Any,
    ):
        self.model_path = model_path
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.chat_format = chat_format
        self.verbose = bool(verbose)
        self._extra_kw = dict(extra_kw)
        self._llm = None          # chat/completion handle
        self._embedder = None     # embedding-mode handle (separate instance)

    # ── handle construction ────────────────────────────────────────────────

    def _build_llama(self, *, embedding: bool = False):
        # Validate BEFORE the deferred import so missing-path errors surface
        # clearly even when llama-cpp-python isn't installed.
        if not self.model_path:
            raise ValueError(
                "LlamaCppClient: model_path is empty — set the GGUF path in the node panel"
            )
        if not os.path.isfile(self.model_path):
            raise RuntimeError(
                f"LlamaCppClient: model file not found: {self.model_path!r}"
            )
        from llama_cpp import Llama  # deferred
        kwargs: dict[str, Any] = {
            "model_path":   self.model_path,
            "n_ctx":        self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose":      self.verbose,
        }
        if embedding:
            kwargs["embedding"] = True
        elif self.chat_format:
            kwargs["chat_format"] = self.chat_format
        kwargs.update(self._extra_kw)
        return Llama(**kwargs)

    def _get_llm(self):
        if self._llm is None:
            self._llm = self._build_llama(embedding=False)
        return self._llm

    def _get_embedder(self):
        if self._embedder is None:
            # A chat-configured Llama *can* embed if embedding=True was set at
            # construction; otherwise we need a second instance. Keep it simple
            # — always build a dedicated embedder for the embed path.
            self._embedder = self._build_llama(embedding=True)
        return self._embedder

    def _model_name(self) -> str:
        return os.path.basename(self.model_path) or "llama-cpp-model"

    @staticmethod
    def _build_kwargs(kw: dict) -> dict:
        """Translate normalized kwargs into llama-cpp's create_chat_completion signature."""
        out: dict[str, Any] = {}
        for k in ("temperature", "max_tokens", "top_p", "top_k", "repeat_penalty",
                  "stop", "seed"):
            if k in kw and kw[k] is not None:
                out[k] = kw[k]
        return out

    # ── LLMClient surface ──────────────────────────────────────────────────

    def chat(
        self, messages: list[Message], *, model: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None, **kw: Any,
    ) -> ChatResult:
        """`model` argument is ignored — the loaded GGUF IS the model."""
        llm = self._get_llm()
        payload: dict[str, Any] = {
            "messages": [msg.to_dict() for msg in messages],
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format
        payload.update(self._build_kwargs(kw))

        t0 = time.time()
        resp = llm.create_chat_completion(**payload)
        elapsed_ms = (time.time() - t0) * 1000.0

        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        out_msg = Message(
            role=msg.get("role", "assistant"),
            content=msg.get("content") or "",
            tool_calls=msg.get("tool_calls") or None,
        )
        usage = resp.get("usage") or {}
        return ChatResult(
            message=out_msg, model=self._model_name(),
            tokens_in=int(usage.get("prompt_tokens", 0) or 0),
            tokens_out=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=elapsed_ms, raw=resp,
        )

    def stream(
        self, messages: list[Message], *, model: str | None = None,
        tools: list[dict] | None = None, **kw: Any,
    ) -> Iterator[str]:
        llm = self._get_llm()
        payload: dict[str, Any] = {
            "messages": [msg.to_dict() for msg in messages],
            "stream":   True,
        }
        if tools:
            payload["tools"] = tools
        payload.update(self._build_kwargs(kw))
        for chunk in llm.create_chat_completion(**payload):
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                yield piece

    def embed(
        self, texts: list[str], *, model: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        emb = self._get_embedder()
        # llama-cpp embed() returns list[float] for a single string, list[list[float]]
        # for a batch. Normalize to list[list[float]] in both cases.
        result = emb.create_embedding(input=list(texts))
        data = result.get("data", []) if isinstance(result, dict) else []
        return [list(item.get("embedding", [])) for item in data]

    def list_models(self) -> list[ModelInfo]:
        """Single-model backend — returns an entry only if the GGUF exists on disk."""
        if not self.model_path or not os.path.isfile(self.model_path):
            return []
        try:
            size = os.path.getsize(self.model_path)
        except OSError:
            size = 0
        return [ModelInfo(name=self._model_name(), size_bytes=size)]

    def ping(self) -> bool:
        """True if the GGUF path exists AND llama_cpp is importable."""
        if not self.model_path or not os.path.isfile(self.model_path):
            return False
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def close(self) -> None:
        """Release both Llama handles. Call before loading a different model."""
        for attr in ("_llm", "_embedder"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)
