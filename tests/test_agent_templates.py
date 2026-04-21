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
    ("templates.agent_research_assistant",
     "Agent Research Assistant (tools + memory + prompt)"),
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
    "templates.agent_research_assistant",
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


def test_agent_autoresearch_mlp_has_ab_cone_and_agent():
    """A/B markers with matching group + a single AutoresearchAgent node."""
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
    # Exactly one AutoresearchAgent on the canvas; no leftover floaters.
    types = [n.type_name for n in g.nodes.values()]
    assert types.count("ag_autoresearch") == 1
    assert "ag_mutator" not in types
    assert "ag_evaluator" not in types
    assert "ag_experiment_loop" not in types


def test_agent_autoresearch_agent_wired_to_llm():
    from templates.agent_autoresearch_mlp import build
    g = Graph()
    build(g)
    agent = next(n for n in g.nodes.values() if n.type_name == "ag_autoresearch")
    has_llm = any(c.to_node_id == agent.id and c.to_port == "llm"
                  for c in g.connections)
    assert has_llm


def test_agent_autoresearch_control_wires_define_search_space():
    """The AutoresearchAgent must have outgoing `control` wires — the
    template encodes the search space as visible canvas wires, not as a
    hidden allowlist string. Each wire targets a real input port on a
    real node in the graph."""
    from templates.agent_autoresearch_mlp import build
    g = Graph()
    build(g)
    agent = next(n for n in g.nodes.values() if n.type_name == "ag_autoresearch")
    control_wires = [c for c in g.connections
                     if c.from_node_id == agent.id and c.from_port == "control"]
    assert len(control_wires) >= 3, (
        f"too few control wires; canvas should show search scope explicitly. "
        f"Got {len(control_wires)}.")
    # Every wire's target port actually exists on the target node.
    for c in control_wires:
        target = g.nodes[c.to_node_id]
        assert c.to_port in target.inputs, (
            f"agent.control → {target.type_name}.{c.to_port} but port doesn't exist")
    # The template should at minimum tune activation and learning rate
    # — shape-safe knobs with big impact on convergence.
    target_ports = {(g.nodes[c.to_node_id].type_name, c.to_port)
                    for c in control_wires}
    assert ("pt_linear", "activation") in target_ports
    assert ("pt_train_marker", "lr") in target_ports


def test_agent_autoresearch_playbook_describes_strategy():
    from templates.agent_autoresearch_mlp import build
    g = Graph()
    build(g)
    agent = next(n for n in g.nodes.values() if n.type_name == "ag_autoresearch")
    pb = (agent.inputs["playbook"].default_value or "").lower()
    # The playbook should hint at which dimensions to explore.
    for keyword in ("activation", "lr"):
        assert keyword in pb, f"playbook missing '{keyword}' guidance"


def test_template_registry_picks_up_agent_templates():
    """The template loader at templates/__init__.py discovers files by
    filename — new agent_*.py files must appear in get_templates()."""
    from templates import get_templates
    labels = {t[0] for t in get_templates()}
    assert "Agent Chat (Ollama + 1 tool)" in labels
    assert "Agent RAG (docs → embed → store → retrieve)" in labels
    assert "Agent Autoresearch (mutate → eval → keep MLP)" in labels
    assert "Agent Research Assistant (tools + memory + prompt)" in labels


def test_research_assistant_has_three_tools_and_prompt_template():
    """The research-assistant template demonstrates tools + memory +
    prompting together — three PythonFunctionTool nodes (two pure, one
    side-effect), a PromptTemplate wired into system_prompt, and the full
    ingestion pipeline."""
    from templates.agent_research_assistant import build
    g = Graph()
    build(g)
    types = [n.type_name for n in g.nodes.values()]
    assert types.count("ag_python_function_tool") == 3
    assert types.count("ag_prompt_template") == 1
    assert types.count("ag_document_loader") == 1
    assert types.count("ag_embedder") == 1
    assert types.count("ag_memory_store") == 1

    # Exactly one side-effect tool (save_note).
    side_tools = [n for n in g.nodes.values()
                  if n.type_name == "ag_python_function_tool"
                  and n.inputs["side_effect"].default_value is True]
    assert len(side_tools) == 1
    assert side_tools[0].inputs["name"].default_value == "save_note"

    # Agent must have allow_side_effect_tools=True for save_note to run.
    agent = next(n for n in g.nodes.values() if n.type_name == "ag_agent")
    assert agent.inputs["allow_side_effect_tools"].default_value is True

    # System prompt is wired from the PromptTemplate, not inline.
    tpl = next(n for n in g.nodes.values() if n.type_name == "ag_prompt_template")
    assert any(c.from_node_id == tpl.id and c.to_node_id == agent.id
               and c.to_port == "system_prompt" for c in g.connections)


def test_research_assistant_tools_compile_and_calc_runs():
    """Each of the three PythonFunctionTools must compile; the pure calc
    tool must return the right answer when invoked directly."""
    from templates.agent_research_assistant import build
    from nodes.agents.python_function_tool import PythonFunctionToolNode
    g = Graph()
    build(g)
    compiled: dict[str, object] = {}
    for n in g.nodes.values():
        if not isinstance(n, PythonFunctionToolNode):
            continue
        name = n.inputs["name"].default_value
        out = n.execute({k: n.inputs[k].default_value
                         for k in ("name", "description", "input_schema",
                                   "code", "side_effect")})
        compiled[name] = out["tool"].callable

    assert set(compiled) == {"search_notes", "calc", "save_note"}
    # Pure calc tool — verify arithmetic works and a hostile expression is rejected.
    assert compiled["calc"](expression="(1 + 2) * 3") == "9"
    assert compiled["calc"](expression="__import__('os').system('x')").startswith("error:")


def test_research_assistant_exports_and_compiles():
    from templates.agent_research_assistant import build
    g = Graph()
    build(g)
    src = GraphExporter().export(g)
    compile(src, "<agent_research_assistant>", "exec")
    # The exported graph uses the shared _ag_chat helper (emitted once).
    assert src.count("def _ag_chat(") == 1
    assert "def main() -> None:" in src


def test_research_assistant_requirements_omit_torch():
    """Hash-64 embedder → no torch or sentence-transformers; qdrant + ollama only."""
    from templates.agent_research_assistant import build
    from plugins.agents._export.requirements import render_requirements
    g = Graph()
    build(g)
    reqs = render_requirements(g)
    assert "ollama" in reqs
    assert "qdrant-client" in reqs
    assert "torch" not in reqs
    assert "sentence-transformers" not in reqs
