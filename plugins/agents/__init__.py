"""Agents plugin — local LLM-driven agent nodes, orchestrator, panel.

Builds, runs, and (eventually) deploys agents using fully open-source local
infrastructure. Default LLM backend: Ollama. Default vector store: Qdrant.
Default embedder: sentence-transformers. See plugins/agents/DESIGN.md.

Phase A + B.1 + B.2-memory (current): LLM clients (Ollama + OpenAI-compat),
prompt / message / conversation primitives, AgentNode with function-calling
tool loop, tool nodes (dotted-path + inline Python), memory nodes
(DocumentLoader, Embedder, MemoryStore, Retriever — Qdrant local default),
AgentOrchestrator with start/stop/state, panel spec. Phases B remaining
(streaming, LlamaCpp, MCP), C (autoresearch), D (deployment) follow.

Heavy imports (ollama, openai, qdrant_client, sentence_transformers,
llama_cpp) are deferred to node-execute time — `register()` runs cleanly
with none of them installed.
"""
from __future__ import annotations
from core.plugins import PluginContext


def register(ctx: PluginContext) -> None:
    """Register all Agents functionality."""
    # Port types first — nodes reference them during _setup_ports.
    from plugins.agents.port_types import register_all as register_port_types
    register_port_types()

    # Auto-discover all node classes from nodes/agents/
    import nodes.agents as ag_pkg
    ctx.discover_nodes(ag_pkg)

    # Top-level palette category. Subcategories ("LLM", "Prompts", "Agent")
    # live on each node class.
    ctx.add_categories(["Agents"])

    # Agents panel spec
    from plugins.agents._panel_agents import build_agents_panel_spec
    ctx.register_panel_spec("Agents", build_agents_panel_spec())

    # RPC orchestrator — agents_orchestrator.handle_rpc routes every method
    # starting with "agent_" or "get_agent_". Lazy: built once per graph.
    def _factory(graph):
        from plugins.agents.agents_orchestrator import AgentOrchestrator
        return AgentOrchestrator(graph)
    ctx.register_orchestrator(["agent_", "get_agent_"], _factory)
