"""End-to-end autoresearch via `agent_autoresearch_*` RPCs.

Orchestrator reads config off the three Phase-C nodes (Mutator, Evaluator,
ExperimentLoop), walks the A/B cone for the group, and drives an
`ExperimentLoop` on a background thread. The training backend is mocked
via a fake OrchestratorRegistry that routes train_* to a canned response.
"""
from __future__ import annotations
import json
import time

import pytest

from core.graph import Graph
from core.node import BaseNode, MarkerRole, PortType
from core.plugins import OrchestratorRegistry
from plugins.agents.agents_orchestrator import AgentOrchestrator


@pytest.fixture(autouse=True)
def _register_port_types():
    from plugins.agents.port_types import register_all
    register_all()


# ── Minimal graph pieces (no torch) ────────────────────────────────────────

class _InputMarker(BaseNode):
    type_name   = "_ar_rpc_input"
    label       = "A"
    marker_role = MarkerRole.INPUT
    def _setup_ports(self):
        self.add_input("group", PortType.STRING, default="task_1")
        self.add_output("tensor", PortType.ANY)
    def execute(self, inputs): return {"tensor": None}


class _TargetMarker(BaseNode):
    type_name   = "_ar_rpc_target"
    label       = "B"
    marker_role = MarkerRole.TRAIN_TARGET
    def _setup_ports(self):
        self.add_input("tensor_in", PortType.ANY)
        self.add_input("group", PortType.STRING, default="task_1")
        self.add_output("config", PortType.ANY)
    def execute(self, inputs): return {"config": {}}


class _Layer(BaseNode):
    type_name = "_ar_rpc_layer"
    label     = "Layer"
    def _setup_ports(self):
        self.add_input("x", PortType.ANY)
        self.add_input("width", PortType.INT, default=32)
        self.add_output("y", PortType.ANY)
    def execute(self, inputs): return {"y": inputs.get("x")}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    from nodes import NODE_REGISTRY
    for cls in (_InputMarker, _TargetMarker, _Layer):
        NODE_REGISTRY[cls.type_name] = cls
    yield
    for cls in (_InputMarker, _TargetMarker, _Layer):
        NODE_REGISTRY.pop(cls.type_name, None)


# ── Fake LLM + fake pytorch backend ───────────────────────────────────────

class _ScriptedMutatorLLM:
    """Returns a canned sequence of mutation JSON blobs."""
    capabilities = {"chat"}
    def __init__(self, responses):
        self._r = list(responses)
        self.calls = 0
    def chat(self, messages, *, model=None, temperature=0.5, **kw):
        from plugins.agents._llm.protocol import ChatResult, Message
        self.calls += 1
        spec = self._r.pop(0) if self._r else '{"op": "remove_node", "node_id": "none"}'
        return ChatResult(
            message=Message(role="assistant", content=spec),
            model=model or "scripted",
        )


class _LLMSource(BaseNode):
    """Tiny helper node that emits a pre-built LLM handle (no network)."""
    type_name = "_ar_rpc_llm_src"
    label     = "LLMSource"
    def __init__(self, llm=None):
        self._llm = llm
        super().__init__()
    def _setup_ports(self):
        self.add_output("llm", "LLM")
    def execute(self, inputs): return {"llm": self._llm}


class _FakeTrainingOrch:
    """Stand-in for TrainingOrchestrator.handle_rpc — canned sequence of
    state transitions + a losses dict."""
    def __init__(self, scores):
        self._scores = list(scores)
        self.calls: list[tuple[str, dict]] = []
    def handle_rpc(self, method, params):
        self.calls.append((method, dict(params or {})))
        if method == "train_start":       return {"ok": True}
        if method == "get_training_state": return {"status": "done"}
        if method == "get_training_losses":
            s = self._scores.pop(0) if self._scores else 0.5
            return {"val_loss": [s]}
        if method == "train_stop": return {"ok": True}
        raise ValueError(f"Unknown: {method}")


def _build_registry(graph, llm, scores):
    """Wire the agents orchestrator + a fake pytorch backend into a registry."""
    fake_train = _FakeTrainingOrch(scores)
    reg = OrchestratorRegistry(graph, [
        (["agent_", "get_agent_"], lambda g: AgentOrchestrator(g)),
        (["train_", "get_training_"], lambda g: fake_train),
    ])
    # Force-resolve agent orch so it receives attach_registry()
    ag = reg.resolve("agent_list_agent_nodes")
    return reg, ag, fake_train


def _build_ar_graph(llm, *, group="task_1", playbook="flip widths",
                    trials=3, wall_clock_s=10.0):
    from nodes.agents.mutator import MutatorNode
    from nodes.agents.evaluator import EvaluatorNode
    from nodes.agents.experiment_loop import ExperimentLoopNode

    g = Graph()
    a = _InputMarker()
    layer = _Layer()
    b = _TargetMarker()
    llm_src = _LLMSource(llm=llm)
    mut = MutatorNode()
    evl = EvaluatorNode()
    lp  = ExperimentLoopNode()

    for n in (a, layer, b, llm_src, mut, evl, lp):
        g.add_node(n)
    a.inputs["group"].default_value = group
    b.inputs["group"].default_value = group
    g.add_connection(a.id, "tensor", layer.id, "x")
    g.add_connection(layer.id, "y", b.id, "tensor_in")
    g.add_connection(llm_src.id, "llm", mut.id, "llm")

    mut.inputs["group"].default_value = group
    mut.inputs["playbook"].default_value = playbook
    mut.inputs["temperature"].default_value = 0.0

    evl.inputs["metric"].default_value = "val_loss"
    evl.inputs["budget_seconds"].default_value = 1.0
    evl.inputs["epochs"].default_value = 1
    evl.inputs["group"].default_value = group

    lp.inputs["trials"].default_value = trials
    lp.inputs["wall_clock_s"].default_value = wall_clock_s
    lp.inputs["loss_threshold"].default_value = 0.0

    return g, mut.id, evl.id, lp.id, layer.id


def _wait_done(agent_orch, run_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = agent_orch.autoresearch_state({"run_id": run_id})
        if st["current_status"] in ("done", "stopped", "error"):
            return st
        time.sleep(0.02)
    raise AssertionError(f"run {run_id!r} did not finish within {timeout}s")


# ── Tests ──────────────────────────────────────────────────────────────────

def test_autoresearch_start_requires_mutator_node():
    orch = AgentOrchestrator(Graph())
    # Even without a registry the missing-node error should fire first.
    resp = orch.autoresearch_start({})
    assert resp["ok"] is False
    assert "MutatorNode" in resp["error"]


def test_autoresearch_start_requires_registry():
    from nodes.agents.mutator import MutatorNode
    from nodes.agents.evaluator import EvaluatorNode
    from nodes.agents.experiment_loop import ExperimentLoopNode
    g = Graph()
    for cls in (_InputMarker, _TargetMarker, MutatorNode, EvaluatorNode,
                ExperimentLoopNode):
        g.add_node(cls())
    orch = AgentOrchestrator(g)   # registry NOT attached
    resp = orch.autoresearch_start({})
    assert resp["ok"] is False
    assert "registry" in resp["error"].lower()


def test_autoresearch_end_to_end_drives_three_trials():
    """Three scripted mutations → three fake trials → best_score picked."""
    llm = _ScriptedMutatorLLM([
        json.dumps({"op": "set_input", "node_id": "__LAYER__",
                    "port": "width", "value": 64}),
        json.dumps({"op": "set_input", "node_id": "__LAYER__",
                    "port": "width", "value": 128}),
        json.dumps({"op": "set_input", "node_id": "__LAYER__",
                    "port": "width", "value": 256}),
    ])
    g, mut_id, evl_id, lp_id, layer_id = _build_ar_graph(llm)
    # Patch the placeholder layer id in the scripted responses
    for i, r in enumerate(llm._r):
        llm._r[i] = r.replace("__LAYER__", layer_id)

    scores = [0.9, 0.4, 0.7]     # trial 2 wins
    reg, agent_orch, fake_train = _build_registry(g, llm, scores)

    resp = agent_orch.handle_rpc("agent_autoresearch_start", {
        "mutator_node_id": mut_id, "evaluator_node_id": evl_id,
        "loop_node_id": lp_id,
    })
    assert resp["ok"] is True
    run_id = resp["run_id"]

    final = _wait_done(agent_orch, run_id)
    assert final["current_status"] == "done"
    assert final["trials_done"] == 3
    assert final["best_score"] == pytest.approx(0.4)
    # Every mutation went to the layer via set_input → layer's current width is
    # the winning trial's value (trial 2 was kept; trial 3 reverted).
    assert g.nodes[layer_id].inputs["width"].default_value == 128


def test_autoresearch_stop_cuts_the_loop_short():
    """Gate the LLM; stop before we release → `current_status == 'stopped'`."""
    import threading
    release = threading.Event()

    class _GatedLLM(_ScriptedMutatorLLM):
        def chat(self, messages, *, model=None, temperature=0.5, **kw):
            release.wait(timeout=2.0)
            return super().chat(messages, model=model, temperature=temperature, **kw)

    llm = _GatedLLM([
        json.dumps({"op": "set_input", "node_id": "__LAYER__",
                    "port": "width", "value": 64})
    ] * 20)
    g, mut_id, evl_id, lp_id, layer_id = _build_ar_graph(
        llm, trials=20, wall_clock_s=30.0,
    )
    for i, r in enumerate(llm._r):
        llm._r[i] = r.replace("__LAYER__", layer_id)

    reg, agent_orch, fake_train = _build_registry(g, llm, [0.5] * 20)
    r = agent_orch.handle_rpc("agent_autoresearch_start", {
        "mutator_node_id": mut_id, "evaluator_node_id": evl_id,
        "loop_node_id": lp_id,
    })
    run_id = r["run_id"]
    # Stop BEFORE releasing the LLM — the first trial never completes.
    stop_resp = agent_orch.handle_rpc("agent_autoresearch_stop", {"run_id": run_id})
    assert stop_resp["ok"] is True
    release.set()
    final = _wait_done(agent_orch, run_id)
    assert final["current_status"] in ("stopped", "done")
    assert final["trials_done"] < 20


def test_autoresearch_state_defaults_to_latest_run():
    orch = AgentOrchestrator(Graph())
    st = orch.autoresearch_state({})
    assert st["trials_done"] == 0
    assert st["current_status"] == "idle"


def test_autoresearch_stop_unknown_run_errors():
    orch = AgentOrchestrator(Graph())
    r = orch.autoresearch_stop({"run_id": "bogus"})
    assert r["ok"] is False


def test_handle_rpc_routes_autoresearch_methods():
    """All three autoresearch methods must route through handle_rpc."""
    orch = AgentOrchestrator(Graph())
    # autoresearch_start/stop return ok=False here because there's no graph,
    # but must not raise ValueError on routing.
    assert "ok" in orch.handle_rpc("agent_autoresearch_start", {})
    assert "trials_done" in orch.handle_rpc("agent_autoresearch_state", {})
    assert "ok" in orch.handle_rpc("agent_autoresearch_stop", {"run_id": "x"})


def test_panel_spec_exposes_autoresearch_sub_section():
    from plugins.agents._panel_agents import build_agents_panel_spec
    spec = build_agents_panel_spec()
    ids = {s.id for s in spec.sections}
    assert "autoresearch_status" in ids
    assert "autoresearch_controls" in ids
