"""GraphAsToolNode — wrap A/B-bounded subgraph as a callable TOOL.

Uses lightweight stand-in markers (pure-Python, no torch) so the test
suite exercises the cone-execution path without dragging pytorch in.
"""
from __future__ import annotations
import pytest

from core.graph import Graph
from core.node import BaseNode, MarkerRole, PortType


@pytest.fixture(autouse=True)
def _register_port_types():
    from plugins.agents.port_types import register_all
    register_all()


# ── Minimal A / B / body markers (no torch) ────────────────────────────────

class _InputMarker(BaseNode):
    """Stand-in for pt_input_marker: has `group` field + stashed probe value."""
    type_name   = "_test_input_marker"
    label       = "A (in)"
    marker_role = MarkerRole.INPUT

    def __init__(self):
        self._probe_tensor = None
        super().__init__()

    def _setup_ports(self):
        self.add_input("group",    PortType.STRING, default="task_1")
        self.add_input("modality", PortType.STRING, default="x")
        self.add_output("tensor",  PortType.ANY)

    def execute(self, inputs):
        return {"tensor": self._probe_tensor}


class _TargetMarker(BaseNode):
    """Stand-in for pt_train_marker."""
    type_name   = "_test_target_marker"
    label       = "B (out)"
    marker_role = MarkerRole.TRAIN_TARGET

    def _setup_ports(self):
        self.add_input("tensor_in", PortType.ANY, default=None)
        self.add_input("group",     PortType.STRING, default="task_1")
        self.add_output("config",   PortType.ANY)

    def execute(self, inputs):
        return {"config": {"tensor_in": inputs.get("tensor_in")}}


class _AddN(BaseNode):
    """Pure-Python body node: y = x + n."""
    type_name = "_test_add_n"
    label     = "AddN"

    def _setup_ports(self):
        self.add_input("x", PortType.ANY)
        self.add_input("n", PortType.INT, default=1)
        self.add_output("y", PortType.ANY)

    def execute(self, inputs):
        x = inputs.get("x")
        n = int(inputs.get("n") or 1)
        return {"y": (x if x is not None else 0) + n}


def _build_graph(group: str = "task_1", n: int = 1):
    g = Graph()
    a = _InputMarker()
    body = _AddN()
    b = _TargetMarker()
    for node in (a, body, b):
        g.add_node(node)
    a.inputs["group"].default_value = group
    body.inputs["n"].default_value = n
    b.inputs["group"].default_value = group
    g.add_connection(a.id, "tensor", body.id, "x")
    g.add_connection(body.id, "y", b.id, "tensor_in")
    return g, a, body, b


# ── Registration ──────────────────────────────────────────────────────────

def test_graph_as_tool_node_registered():
    from core.plugins import PluginContext
    import plugins.agents as agents_pkg
    ctx = PluginContext()
    agents_pkg.register(ctx)
    names = {c.type_name for c in ctx.node_classes}
    assert "ag_graph_as_tool" in names


# ── Standalone execute fails cleanly ───────────────────────────────────────

def test_standalone_execute_raises():
    from nodes.agents.graph_as_tool import GraphAsToolNode
    n = GraphAsToolNode()
    # No _graph attribute → clear error
    with pytest.raises(RuntimeError, match="live Graph"):
        n.execute({"group": "task_1", "name": "x", "description": "",
                   "side_effect": False})


# ── Building the tool inside a graph ───────────────────────────────────────

def test_execute_produces_tool_with_schema_from_a_outputs():
    from nodes.agents.graph_as_tool import GraphAsToolNode
    g, a, body, b = _build_graph()
    tool_node = GraphAsToolNode()
    g.add_node(tool_node)
    tool_node.inputs["name"].default_value = "run_task"
    tool_node.inputs["description"].default_value = "Run the task"
    tool_node.inputs["group"].default_value = "task_1"

    stored, _, errors = g.execute()
    assert errors == {}
    td = stored[tool_node.id]["tool"]
    assert td.name == "run_task"
    assert td.description == "Run the task"
    # Schema declares one property per A-output port; A has "tensor".
    assert td.input_schema["required"] == ["tensor"]
    assert "tensor" in td.input_schema["properties"]


def test_tool_callable_runs_subgraph_and_returns_b_output():
    from nodes.agents.graph_as_tool import GraphAsToolNode
    g, a, body, b = _build_graph(n=10)
    tool_node = GraphAsToolNode()
    g.add_node(tool_node)

    stored, _, _ = g.execute()
    td = stored[tool_node.id]["tool"]

    # Invoking the tool with tensor=5 → body emits 5+10=15 → B sees tensor_in=15
    out = td.callable(tensor=5)
    assert out == {"config": {"tensor_in": 15}}


def test_tool_callable_covers_multiple_body_nodes():
    """Two AddN nodes in series: 7 + 3 + 4 = 14."""
    from nodes.agents.graph_as_tool import GraphAsToolNode

    g = Graph()
    a = _InputMarker()
    n1 = _AddN()
    n2 = _AddN()
    b = _TargetMarker()
    for node in (a, n1, n2, b):
        g.add_node(node)
    n1.inputs["n"].default_value = 3
    n2.inputs["n"].default_value = 4
    g.add_connection(a.id, "tensor", n1.id, "x")
    g.add_connection(n1.id, "y", n2.id, "x")
    g.add_connection(n2.id, "y", b.id, "tensor_in")

    tool_node = GraphAsToolNode()
    g.add_node(tool_node)
    stored, _, _ = g.execute()
    td = stored[tool_node.id]["tool"]

    out = td.callable(tensor=7)
    assert out["config"]["tensor_in"] == 14


def test_tool_honors_group_selector():
    """Two A/B pairs with different groups; the node's `group` field picks one."""
    from nodes.agents.graph_as_tool import GraphAsToolNode

    g = Graph()
    a1, a2 = _InputMarker(), _InputMarker()
    body1, body2 = _AddN(), _AddN()
    b1, b2 = _TargetMarker(), _TargetMarker()
    for node in (a1, a2, body1, body2, b1, b2):
        g.add_node(node)
    a1.inputs["group"].default_value = "alpha"
    a2.inputs["group"].default_value = "beta"
    b1.inputs["group"].default_value = "alpha"
    b2.inputs["group"].default_value = "beta"
    body1.inputs["n"].default_value = 100
    body2.inputs["n"].default_value = 200
    g.add_connection(a1.id, "tensor", body1.id, "x")
    g.add_connection(body1.id, "y", b1.id, "tensor_in")
    g.add_connection(a2.id, "tensor", body2.id, "x")
    g.add_connection(body2.id, "y", b2.id, "tensor_in")

    # Pick group=beta
    tool_node = GraphAsToolNode()
    g.add_node(tool_node)
    tool_node.inputs["group"].default_value = "beta"
    stored, _, _ = g.execute()
    td = stored[tool_node.id]["tool"]

    out = td.callable(tensor=0)
    assert out["config"]["tensor_in"] == 200   # used body2 (n=200)


def test_tool_missing_marker_raises():
    """No B marker → execute surfaces a clear error as a graph-execute error."""
    from nodes.agents.graph_as_tool import GraphAsToolNode

    g = Graph()
    a = _InputMarker()
    g.add_node(a)
    tool_node = GraphAsToolNode()
    g.add_node(tool_node)

    stored, _, errors = g.execute()
    assert tool_node.id in errors
    assert "TRAIN_TARGET" in errors[tool_node.id]["message"]


def test_tool_side_effect_flag_propagates():
    from nodes.agents.graph_as_tool import GraphAsToolNode
    g, a, body, b = _build_graph()
    tool_node = GraphAsToolNode()
    g.add_node(tool_node)
    tool_node.inputs["side_effect"].default_value = True
    stored, _, _ = g.execute()
    td = stored[tool_node.id]["tool"]
    assert td.side_effect is True


def test_tool_via_agent_function_calling_loop():
    """End-to-end: GraphAsTool plugs into AgentNode's tool slots."""
    from nodes.agents.graph_as_tool import GraphAsToolNode
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.protocol import Message

    g, a, body, b = _build_graph(n=50)
    tool_node = GraphAsToolNode()
    g.add_node(tool_node)
    tool_node.inputs["name"].default_value = "compute"
    stored, _, _ = g.execute()
    td = stored[tool_node.id]["tool"]

    class _ScriptedLLM:
        capabilities = {"tools"}
        def __init__(self, responses):
            self._r = list(responses)
            self.calls = 0
        def chat(self, messages, *, model=None, tools=None, **kw):
            from plugins.agents._llm.protocol import ChatResult
            self.calls += 1
            spec = self._r.pop(0)
            return ChatResult(
                message=Message(role="assistant",
                                content=spec.get("content", ""),
                                tool_calls=spec.get("tool_calls")),
                model="scripted",
            )

    llm = _ScriptedLLM([
        {"tool_calls": [{"function": {"name": "compute",
                                       "arguments": {"tensor": 2}}}]},
        {"content": "done"},
    ])
    agent = AgentNode()
    out = agent.execute({
        "llm": llm, "messages": [Message(role="user", content="go")],
        "system_prompt": "", "model": "", "temperature": 0.0,
        "tool_1": td, "tool_2": None, "tool_3": None, "tool_4": None,
        "max_iterations": 3, "allow_side_effect_tools": False,
    })
    assert out["text"] == "done"
    tc = out["tool_calls"][0]
    assert tc["name"] == "compute"
    # 2 + 50 = 52 → B emits config with tensor_in=52
    assert tc["result"]["config"]["tensor_in"] == 52
