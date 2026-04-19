"""AgentOrchestrator handle_rpc dispatch + a session run via MockLLMClient."""
import pytest


@pytest.fixture(autouse=True)
def _register_agent_port_types():
    from plugins.agents.port_types import register_all
    register_all()


def test_handle_rpc_unknown_raises():
    from core.graph import Graph
    from plugins.agents.agents_orchestrator import AgentOrchestrator
    orch = AgentOrchestrator(Graph())
    with pytest.raises(ValueError):
        orch.handle_rpc("nope", {})


def test_handle_rpc_routes_known_methods():
    from core.graph import Graph
    from plugins.agents.agents_orchestrator import AgentOrchestrator
    orch = AgentOrchestrator(Graph())

    assert orch.handle_rpc("agent_list_agent_nodes", {}) == {"items": []}
    drain = orch.handle_rpc("agent_drain_tokens", {})
    assert drain["done"] is True and drain["chunks"] == []
    state = orch.handle_rpc("get_agent_state", {})
    assert state["status"] == "Idle"


def test_list_local_models_handles_missing_backend():
    """With no Ollama daemon reachable, list_local_models should return
    ok=False with an error string — not raise."""
    from core.graph import Graph
    from plugins.agents.agents_orchestrator import AgentOrchestrator
    orch = AgentOrchestrator(Graph())
    out = orch.handle_rpc("agent_list_local_models", {})
    assert "items" in out
    assert "host" in out
    assert "ok" in out


def test_agent_start_with_mock_llm(monkeypatch):
    """Build a graph with one OllamaClientNode wired to one AgentNode, but
    monkey-patch the resolver so it returns a MockLLMClient."""
    from core.graph import Graph, Connection
    from nodes.agents.ollama_client import OllamaClientNode
    from nodes.agents.agent import AgentNode
    from plugins.agents.agents_orchestrator import AgentOrchestrator
    from plugins.agents._llm.mock_client import MockLLMClient

    g = Graph()
    cli = OllamaClientNode()
    ag  = AgentNode()
    g.nodes[cli.id] = cli
    g.nodes[ag.id]  = ag
    g.connections.append(Connection(cli.id, "llm", ag.id, "llm"))

    orch = AgentOrchestrator(g)
    monkeypatch.setattr(
        orch, "_resolve_llm_for_agent",
        lambda node: MockLLMClient(response="hi back"),
    )

    out = orch.handle_rpc("agent_start",
                          {"message": "hello", "agent_id": ag.id})
    assert out["ok"] is True
    assert out["reply"] == "hi back"
    assert out["tokens_out"] == 5

    # State should reflect the just-finished session
    state = orch.handle_rpc("get_agent_state", {})
    assert state["status"] == "Done"
    assert state["reply"] == "hi back"


def test_agent_start_without_agent_node():
    from core.graph import Graph
    from plugins.agents.agents_orchestrator import AgentOrchestrator
    orch = AgentOrchestrator(Graph())
    out = orch.handle_rpc("agent_start", {"message": "hi"})
    assert out["ok"] is False


def test_agent_start_empty_message():
    from core.graph import Graph
    from nodes.agents.agent import AgentNode
    from plugins.agents.agents_orchestrator import AgentOrchestrator
    g = Graph()
    ag = AgentNode()
    g.nodes[ag.id] = ag
    orch = AgentOrchestrator(g)
    out = orch.handle_rpc("agent_start", {"message": "", "agent_id": ag.id})
    assert out["ok"] is False
    assert "Empty message" in out["error"]


def test_agent_start_no_llm_connected():
    from core.graph import Graph
    from nodes.agents.agent import AgentNode
    from plugins.agents.agents_orchestrator import AgentOrchestrator
    g = Graph()
    ag = AgentNode()
    g.nodes[ag.id] = ag
    orch = AgentOrchestrator(g)
    out = orch.handle_rpc("agent_start", {"message": "hi", "agent_id": ag.id})
    assert out["ok"] is False
    assert "No LLM client" in out["error"]


def test_drain_logs_yields_lines_then_resets():
    from core.graph import Graph
    from plugins.agents.agents_orchestrator import AgentOrchestrator
    orch = AgentOrchestrator(Graph())
    orch._pending_logs.extend(["one", "two"])
    out = orch.handle_rpc("agent_drain_logs", {})
    assert out["lines"] == ["one", "two"]
    out2 = orch.handle_rpc("agent_drain_logs", {})
    assert out2["lines"] == []
