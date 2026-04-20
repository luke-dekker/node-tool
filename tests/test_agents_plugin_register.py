"""Verify the agents plugin loads cleanly with no heavy deps installed.

Crucial Phase-A invariant: `import plugins.agents` and `register(ctx)` must
both succeed even with ollama, openai, qdrant_client, sentence_transformers
all absent. Heavy imports happen only inside node `execute()` paths.
"""
import pytest


def test_register_runs_clean():
    """register(ctx) populates port types, nodes, panel spec, categories."""
    from core.plugins import PluginContext
    import plugins.agents as agents_pkg

    ctx = PluginContext()
    agents_pkg.register(ctx)

    type_names = {c.type_name for c in ctx.node_classes}
    assert "ag_ollama_client" in type_names
    assert "ag_openai_compat_client" in type_names
    assert "ag_prompt_template" in type_names
    assert "ag_chat_message" in type_names
    assert "ag_conversation" in type_names
    assert "ag_agent" in type_names

    assert any(label == "Agents" for label, _ in ctx.panel_specs)
    assert "Agents" in ctx.categories


def test_port_types_registered():
    from plugins.agents.port_types import register_all
    from core.port_types import PortTypeRegistry
    register_all()
    for t in ("LLM", "MESSAGE", "CONVERSATION", "PROMPT_TEMPLATE"):
        assert PortTypeRegistry.get(t) is not None, f"Port type {t} not registered"


def test_protocol_module_imports_without_backends():
    """Importing the protocol module must not pull ollama/openai."""
    import plugins.agents._llm.protocol as p
    assert hasattr(p, "Message")
    assert hasattr(p, "ChatResult")
    assert hasattr(p, "ModelInfo")
    assert hasattr(p, "LLMClient")


def test_ollama_client_constructible_without_ollama():
    """The OllamaClient class must be constructible even with no `ollama` pkg.
    Only methods that hit the daemon should require the import."""
    from plugins.agents._llm.ollama_client import OllamaClient
    cli = OllamaClient(host="http://invalid:0", default_model="x")
    assert cli.host == "http://invalid:0"
    assert cli.default_model == "x"
    assert cli._client is None  # not initialized until first call


def test_panel_spec_serializable():
    """PanelSpec must convert to JSON-safe dict (used by RPC transport)."""
    from plugins.agents._panel_agents import build_agents_panel_spec
    spec = build_agents_panel_spec()
    d = spec.to_dict()
    assert d["label"] == "Agents"
    section_ids = {s["id"] for s in d["sections"]}
    assert {"backend", "models", "agents", "chat", "status", "controls"} <= section_ids
    # Chat is the streaming CustomSection wired to agent_start_stream.
    chat = next(s for s in d["sections"] if s["id"] == "chat")
    assert chat["kind"] == "custom"
    assert chat["custom_kind"] == "chat_stream"
    assert chat["params"]["start_rpc"] == "agent_start_stream"
    assert chat["params"]["drain_rpc"] == "agent_drain_tokens"
    # The chat section declares a message input so ButtonsSection.collect
    # can gather it like a FormSection field.
    assert any(f["id"] == "message" for f in chat.get("fields", []))
