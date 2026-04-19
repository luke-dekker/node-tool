"""Agents plugin — local LLM-driven agent nodes, orchestrator, panel.

Builds, runs, and (eventually) deploys agents using fully open-source local
infrastructure. Default LLM backend: Ollama. Default vector store: Qdrant.
Default embedder: sentence-transformers. See plugins/agents/DESIGN.md.

Phase A scope (current): LLM clients (Ollama + OpenAI-compat), prompt /
message / conversation primitives, AgentNode (single chat() turn — no
tools, no streaming yet), AgentOrchestrator with start/stop/state, panel
spec. Phases B (tools+memory+streaming), C (autoresearch), D (deployment)
follow.

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
