"""LlamaCppClient + LlamaCppClientNode — in-process GGUF backend.

Heavy-dep parts (actually loading a GGUF) are skipped unless a model path is
provided via env var. The main invariant is: register + construction + all
parsing logic work with neither llama_cpp installed nor a model on disk.

To run the real-model test locally, set LLAMA_CPP_TEST_MODEL to a GGUF path
(e.g., a small `qwen2.5-0.5b-instruct-q4_k_m.gguf`) and install llama-cpp-python.
"""
from __future__ import annotations
import importlib.util
import os

import pytest


HAS_LLAMA_CPP = importlib.util.find_spec("llama_cpp") is not None
MODEL_PATH = os.environ.get("LLAMA_CPP_TEST_MODEL", "")


@pytest.fixture(autouse=True)
def _register_agent_port_types():
    from plugins.agents.port_types import register_all
    register_all()


# ── register + import invariants ──────────────────────────────────────────

def test_llama_cpp_module_imports_without_backend():
    """Importing the client module must not pull llama_cpp."""
    import plugins.agents._llm.llama_cpp_client as mod
    assert hasattr(mod, "LlamaCppClient")


def test_llama_cpp_client_constructible_without_llama_cpp():
    from plugins.agents._llm.llama_cpp_client import LlamaCppClient
    c = LlamaCppClient(model_path="Z:/does/not/exist.gguf",
                       n_ctx=1024, n_gpu_layers=0)
    assert c.model_path == "Z:/does/not/exist.gguf"
    assert c.n_ctx == 1024
    assert c.n_gpu_layers == 0
    assert c._llm is None      # lazy — nothing loaded yet
    assert c._embedder is None


def test_plugin_registers_llama_cpp_node():
    from core.plugins import PluginContext
    import plugins.agents as agents_pkg
    ctx = PluginContext()
    agents_pkg.register(ctx)
    names = {c.type_name for c in ctx.node_classes}
    assert "ag_llama_cpp_client" in names


# ── Node smoke ─────────────────────────────────────────────────────────────

def test_llama_cpp_node_emits_llm_handle():
    from nodes.agents.llama_cpp_client import LlamaCppClientNode
    out = LlamaCppClientNode().execute({
        "model_path": "Z:/does/not/exist.gguf",
        "n_ctx": 512,
        "n_gpu_layers": 0,
        "chat_format": "",
        "verbose": False,
    })
    llm = out["llm"]
    assert llm.model_path == "Z:/does/not/exist.gguf"
    assert llm.n_ctx == 512
    assert llm.chat_format is None


def test_llama_cpp_node_passes_chat_format():
    from nodes.agents.llama_cpp_client import LlamaCppClientNode
    out = LlamaCppClientNode().execute({
        "model_path": "Z:/m.gguf", "n_ctx": 2048, "n_gpu_layers": -1,
        "chat_format": "chatml-function-calling", "verbose": True,
    })
    llm = out["llm"]
    assert llm.chat_format == "chatml-function-calling"
    assert llm.verbose is True
    assert llm.n_gpu_layers == -1


# ── list_models / ping without a model file ───────────────────────────────

def test_list_models_empty_when_file_missing():
    from plugins.agents._llm.llama_cpp_client import LlamaCppClient
    c = LlamaCppClient(model_path="Z:/nope.gguf")
    assert c.list_models() == []


def test_list_models_emits_name_and_size_when_file_present(tmp_path):
    from plugins.agents._llm.llama_cpp_client import LlamaCppClient
    fake_gguf = tmp_path / "fake.gguf"
    fake_gguf.write_bytes(b"x" * 1024)
    c = LlamaCppClient(model_path=str(fake_gguf))
    models = c.list_models()
    assert len(models) == 1
    assert models[0].name == "fake.gguf"
    assert models[0].size_bytes == 1024


def test_ping_false_when_file_missing():
    from plugins.agents._llm.llama_cpp_client import LlamaCppClient
    c = LlamaCppClient(model_path="Z:/nope.gguf")
    assert c.ping() is False


def test_chat_missing_model_path_raises():
    from plugins.agents._llm.llama_cpp_client import LlamaCppClient
    from plugins.agents._llm.protocol import Message
    c = LlamaCppClient(model_path="")
    with pytest.raises(ValueError, match="model_path is empty"):
        c.chat([Message(role="user", content="hi")])


def test_chat_nonexistent_file_raises(tmp_path):
    from plugins.agents._llm.llama_cpp_client import LlamaCppClient
    from plugins.agents._llm.protocol import Message
    c = LlamaCppClient(model_path=str(tmp_path / "ghost.gguf"))
    with pytest.raises(RuntimeError, match="model file not found"):
        c.chat([Message(role="user", content="hi")])


def test_empty_embed_short_circuits():
    from plugins.agents._llm.llama_cpp_client import LlamaCppClient
    c = LlamaCppClient(model_path="Z:/nope.gguf")
    # Empty input returns [] WITHOUT trying to load the model
    assert c.embed([]) == []


# ── Stub-Llama routing (covers parsing without needing llama_cpp) ──────────

class _StubLlama:
    """Behaves like a `Llama` object — covers chat / stream / embed parsing
    without actually loading a GGUF."""

    def __init__(self, *, chat_response=None, stream_chunks=None,
                 embed_vectors=None):
        self._chat_response = chat_response
        self._stream_chunks = stream_chunks or []
        self._embed_vectors = embed_vectors or []
        self.closed = False

    def create_chat_completion(self, **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            return iter([
                {"choices": [{"delta": {"content": piece}}]}
                for piece in self._stream_chunks
            ])
        return self._chat_response

    def create_embedding(self, *, input):
        return {"data": [{"embedding": vec} for vec in self._embed_vectors]}

    def close(self):
        self.closed = True


def _install_stub(monkeypatch, client, *, chat_response=None,
                  stream_chunks=None, embed_vectors=None):
    chat_stub = _StubLlama(chat_response=chat_response,
                           stream_chunks=stream_chunks)
    emb_stub = _StubLlama(embed_vectors=embed_vectors)
    def fake_build(self, *, embedding=False):
        return emb_stub if embedding else chat_stub
    monkeypatch.setattr(type(client), "_build_llama", fake_build)
    return chat_stub, emb_stub


def test_chat_routes_through_create_chat_completion(monkeypatch):
    from plugins.agents._llm.llama_cpp_client import LlamaCppClient
    from plugins.agents._llm.protocol import Message
    c = LlamaCppClient(model_path="/fake/m.gguf",
                       n_ctx=1024, chat_format="chatml-function-calling")
    chat_stub, _ = _install_stub(monkeypatch, c, chat_response={
        "choices": [{"message": {"role": "assistant", "content": "hi there",
                                   "tool_calls": None}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    })
    out = c.chat([Message(role="user", content="hello")],
                 temperature=0.25, max_tokens=64)
    assert out.message.content == "hi there"
    assert out.tokens_in == 7
    assert out.tokens_out == 3
    assert out.model == "m.gguf"
    # Normalized kwargs translated to llama.cpp signature
    assert chat_stub.last_kwargs["temperature"] == 0.25
    assert chat_stub.last_kwargs["max_tokens"] == 64
    # Messages serialized via Message.to_dict
    assert chat_stub.last_kwargs["messages"] == [
        {"role": "user", "content": "hello"},
    ]


def test_chat_surfaces_tool_calls(monkeypatch):
    from plugins.agents._llm.llama_cpp_client import LlamaCppClient
    from plugins.agents._llm.protocol import Message
    c = LlamaCppClient(model_path="/fake/m.gguf")
    tool_calls = [{"id": "call_x", "function": {"name": "ping", "arguments": "{}"}}]
    _install_stub(monkeypatch, c, chat_response={
        "choices": [{"message": {"role": "assistant", "content": "",
                                   "tool_calls": tool_calls}}],
    })
    out = c.chat([Message(role="user", content="ping")])
    assert out.message.tool_calls == tool_calls


def test_stream_yields_content_deltas(monkeypatch):
    from plugins.agents._llm.llama_cpp_client import LlamaCppClient
    from plugins.agents._llm.protocol import Message
    c = LlamaCppClient(model_path="/fake/m.gguf")
    _install_stub(monkeypatch, c,
                  stream_chunks=["Hello", ", ", "world", "!"])
    pieces = list(c.stream([Message(role="user", content="hi")]))
    assert pieces == ["Hello", ", ", "world", "!"]


def test_embed_returns_aligned_vectors(monkeypatch):
    from plugins.agents._llm.llama_cpp_client import LlamaCppClient
    c = LlamaCppClient(model_path="/fake/m.gguf")
    _install_stub(monkeypatch, c,
                  embed_vectors=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    out = c.embed(["a", "b", "c"])
    assert out == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]


def test_close_releases_both_handles(monkeypatch):
    from plugins.agents._llm.llama_cpp_client import LlamaCppClient
    from plugins.agents._llm.protocol import Message
    c = LlamaCppClient(model_path="/fake/m.gguf")
    chat_stub, emb_stub = _install_stub(monkeypatch, c,
        chat_response={"choices":[{"message":{"role":"assistant","content":"x"}}]},
        embed_vectors=[[0.0]])
    c.chat([Message(role="user", content="x")])
    c.embed(["y"])
    assert c._llm is not None and c._embedder is not None
    c.close()
    assert c._llm is None
    assert c._embedder is None
    assert chat_stub.closed and emb_stub.closed


# ── Real-model integration (opt-in) ────────────────────────────────────────

@pytest.mark.skipif(not HAS_LLAMA_CPP or not MODEL_PATH,
                    reason="llama-cpp-python not installed or LLAMA_CPP_TEST_MODEL unset")
def test_real_chat_round_trip():
    from plugins.agents._llm.llama_cpp_client import LlamaCppClient
    from plugins.agents._llm.protocol import Message
    c = LlamaCppClient(model_path=MODEL_PATH, n_ctx=512, n_gpu_layers=0)
    try:
        out = c.chat([Message(role="user", content="Say exactly: OK")],
                     temperature=0.0, max_tokens=8)
        assert out.message.role == "assistant"
        assert out.message.content.strip()
    finally:
        c.close()
