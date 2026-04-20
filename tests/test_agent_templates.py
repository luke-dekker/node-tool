"""Smoke tests for the three demo templates.

Each template must build a valid graph (no broken connections, no missing
node types) and export-compile cleanly. Running the graph end-to-end
requires Ollama / training / qdrant, which we don't assume; the smoke
check exercises construction + export only.
"""
from __future__ import annotations
import importlib
import importlib.util

import pytest

from core.graph import Graph
from core.exporter import GraphExporter


HAS_QDRANT = importlib.util.find_spec("qdrant_client") is not None


@pytest.fixture(autouse=True)
def _register_port_types():
    from plugins.agents.port_types import register_all
    register_all()


@pytest.mark.parametrize("mod_name,expected_label", [
    ("templates.agent_chat", "Agent Chat (Ollama + 1 tool)"),
    ("templates.agent_rag", "Agent RAG (docs → embed → store → retrieve)"),
    ("templates.agent_autoresearch_mlp",
     "Agent Autoresearch (mutate → eval → keep MLP)"),
])
def test_template_loads_and_has_metadata(mod_name, expected_label):
    mod = importlib.import_module(mod_name)
    assert mod.LABEL == expected_label
    assert isinstance(mod.DESCRIPTION, str) and mod.DESCRIPTION
    assert callable(mod.build)


@pytest.mark.parametrize("mod_name", [
    "templates.agent_chat",
    "templates.agent_rag",
    "templates.agent_autoresearch_mlp",
])
def test_template_builds_a_valid_graph(mod_name):
    mod = importlib.import_module(mod_name)
    g = Graph()
    positions = mod.build(g)
    assert len(g.nodes) > 0
    assert len(positions) == len(g.nodes)
    # Every connection has both endpoints in the graph
    for c in g.connections:
        assert c.from_node_id in g.nodes
        assert c.to_node_id   in g.nodes
        assert c.from_port in g.nodes[c.from_node_id].outputs
        assert c.to_port   in g.nodes[c.to_node_id].inputs
    # Topological order covers all nodes → no cycles
    order = g.topological_order()
    assert len(order) == len(g.nodes)


def test_agent_chat_has_expected_shape():
    from templates.agent_chat import build
    g = Graph()
    build(g)
    types = sorted(n.type_name for n in g.nodes.values())
    assert types == sorted([
        "ag_ollama_client", "ag_chat_message", "ag_conversation",
        "ag_python_function_tool", "ag_agent",
    ])


def test_agent_chat_exports_and_compiles():
    from templates.agent_chat import build
    g = Graph()
    build(g)
    src = GraphExporter().export(g)
    compile(src, "<agent_chat>", "exec")
    # The helper is emitted and there's a main()
    assert "def _ag_chat" in src
    assert "def main() -> None:" in src


def test_agent_chat_requirements_only_ollama():
    from templates.agent_chat import build
    from plugins.agents._export.requirements import render_requirements
    g = Graph()
    build(g)
    reqs = render_requirements(g)
    assert "ollama" in reqs
    # §G.3 invariant
    assert "torch" not in reqs
    assert "qdrant-client" not in reqs
    assert "sentence-transformers" not in reqs


def test_agent_rag_has_memory_pipeline():
    from templates.agent_rag import build
    g = Graph()
    build(g)
    types = {n.type_name for n in g.nodes.values()}
    assert {"ag_document_loader", "ag_embedder",
            "ag_memory_store", "ag_retriever"} <= types


def test_agent_rag_requirements_omit_torch_with_hash_embedder():
    """Default embedder model is `hash-64` → no sentence-transformers/torch."""
    from templates.agent_rag import build
    from plugins.agents._export.requirements import render_requirements
    g = Graph()
    build(g)
    reqs = render_requirements(g)
    assert "qdrant-client" in reqs
    assert "sentence-transformers" not in reqs
    assert "torch" not in reqs


def test_agent_autoresearch_mlp_has_ab_cone():
    """A/B markers with matching group + the three Phase-C nodes."""
    from templates.agent_autoresearch_mlp import build
    from core.node import MarkerRole
    g = Graph()
    build(g)
    a = [n for n in g.nodes_by_role(MarkerRole.INPUT)
         if n.inputs["group"].default_value == "mlp"]
    b = [n for n in g.nodes_by_role(MarkerRole.TRAIN_TARGET)
         if n.inputs["group"].default_value == "mlp"]
    assert len(a) == 1 and len(b) == 1
    cone = g.subgraph_between(a[0].id, b[0].id)
    # A + flatten + 2 linears + B
    assert len(cone) == 5
    # The three autoresearch nodes are present
    types = {n.type_name for n in g.nodes.values()}
    assert {"ag_mutator", "ag_evaluator", "ag_experiment_loop"} <= types


def test_agent_autoresearch_mutator_wired_to_llm():
    from templates.agent_autoresearch_mlp import build
    g = Graph()
    build(g)
    mut = next(n for n in g.nodes.values() if n.type_name == "ag_mutator")
    # Someone connects to mutator's llm input
    has_llm = any(c.to_node_id == mut.id and c.to_port == "llm"
                  for c in g.connections)
    assert has_llm


def test_template_registry_picks_up_agent_templates():
    """The template loader at templates/__init__.py discovers files by
    filename — new agent_*.py files must appear in get_templates()."""
    from templates import get_templates
    labels = {t[0] for t in get_templates()}
    assert "Agent Chat (Ollama + 1 tool)" in labels
    assert "Agent RAG (docs → embed → store → retrieve)" in labels
    assert "Agent Autoresearch (mutate → eval → keep MLP)" in labels
