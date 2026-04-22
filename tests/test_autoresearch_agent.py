"""End-to-end tests for the new autoresearch design.

Covers:
  - `AutoresearchAgentNode` shape (ports + initial state).
  - `collect_targets` walks outgoing `control` wires.
  - `parse_changes` tolerates the LLM responses we expect.
  - `apply_changes` enforces port type, choices, min/max.
  - `ControlLoop` runs end-to-end with a stubbed LLM + fake training orch.
  - `autoresearch_start` RPC validation: missing agent / LLM / wires /
    cached training params each surface a clear error.
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


# ── Stubs ──────────────────────────────────────────────────────────────────

class _ScriptedLLM:
    """LLM stub that returns one queued response per `chat()` call."""
    def __init__(self, responses: list[str]):
        self._r = list(responses)
        self.calls: list[list] = []
    def chat(self, messages, *, model=None, temperature=0.5, **kw):
        from plugins.agents._llm.protocol import Message, ChatResult
        self.calls.append(list(messages))
        body = self._r.pop(0) if self._r else "{}"
        return ChatResult(message=Message(role="assistant", content=body))
    def stream(self, *a, **k):
        raise NotImplementedError


_DEFAULT_TRAIN_PARAMS = {
    "device": "cpu",
    "datasets": {"task_1": {"path": "stub", "batch_size": 8}},
}
_UNSET = object()


class _FakeTrainOrch:
    """Stand-in for TrainingOrchestrator's RPC surface."""
    def __init__(self, scores, last_params=_UNSET):
        self._scores = list(scores)
        self.calls: list = []
        # Distinguish "use default" (constructor caller didn't supply) from
        # "explicit None" (caller wants to test the no-cache path).
        self._last = _DEFAULT_TRAIN_PARAMS if last_params is _UNSET else last_params
    def handle_rpc(self, method, params):
        self.calls.append((method, dict(params or {})))
        if method == "get_training_last_params":
            return {"params": self._last}
        if method == "train_start":
            return {"ok": True}
        if method == "get_training_state":
            return {"status": "done"}
        if method == "get_training_losses":
            s = self._scores.pop(0) if self._scores else 0.5
            return {"series": {"train": [s], "val": [s]}}
        if method == "train_stop":
            return {"ok": True}
        raise ValueError(f"unknown: {method}")


class _Widget(BaseNode):
    """Tiny non-pytorch node so tests don't drag torch into the agent path."""
    type_name = "_t_widget"
    def _setup_ports(self):
        self.add_input("tensor_in", PortType.ANY)
        self.add_input("width",      PortType.INT,    default=8)
        self.add_input("activation", PortType.STRING, default="relu",
                       choices=["relu", "gelu", "tanh"])
        self.add_input("rate",       PortType.FLOAT,  default=0.1)
        self.add_input("freeze",     PortType.BOOL,   default=False)
        self.add_output("tensor_out", PortType.ANY)
    def execute(self, inputs):
        return {"tensor_out": inputs.get("tensor_in")}


class _InputMarker(BaseNode):
    type_name = "_t_input_marker"
    marker_role = MarkerRole.INPUT
    def _setup_ports(self):
        self.add_input("group", PortType.STRING, default="task_1")
        self.add_output("tensor", PortType.ANY)
    def execute(self, inputs): return {"tensor": None}


class _TargetMarker(BaseNode):
    type_name = "_t_target_marker"
    marker_role = MarkerRole.TRAIN_TARGET
    def _setup_ports(self):
        self.add_input("group", PortType.STRING, default="task_1")
        self.add_input("tensor_in", PortType.ANY, default=None)
        self.add_output("config", PortType.ANY)
    def execute(self, inputs): return {"config": {}}


def _build_graph(llm=None):
    """Build A → Widget → B + AutoresearchAgent wired to the widget's
    width, activation, and rate."""
    from nodes.agents.autoresearch_agent import AutoresearchAgentNode
    g = Graph()
    a = _InputMarker();    g.add_node(a)
    w = _Widget();         g.add_node(w)
    b = _TargetMarker();   g.add_node(b)
    g.add_connection(a.id, "tensor",     w.id, "tensor_in")
    g.add_connection(w.id, "tensor_out", b.id, "tensor_in")

    agent = AutoresearchAgentNode()
    agent.inputs["trials"].default_value      = 2
    agent.inputs["wall_clock_s"].default_value = 5.0
    agent.inputs["eval_budget_s"].default_value = 1.0
    g.add_node(agent)

    if llm is not None:
        class _LLMNode(BaseNode):
            type_name = "_t_llm"
            def _setup_ports(self):
                self.add_output("llm", "LLM")
            def execute(self, inputs): return {"llm": llm}
        cli = _LLMNode(); g.add_node(cli)
        g.add_connection(cli.id, "llm", agent.id, "llm")

    g.add_connection(agent.id, "control", w.id, "width")
    g.add_connection(agent.id, "control", w.id, "activation")
    g.add_connection(agent.id, "control", w.id, "rate")
    return g, a, w, b, agent


# ── Node shape ─────────────────────────────────────────────────────────────

def test_autoresearch_agent_node_has_expected_ports():
    from nodes.agents.autoresearch_agent import AutoresearchAgentNode
    n = AutoresearchAgentNode()
    for p in ("llm", "playbook", "group", "metric", "trials",
              "wall_clock_s", "eval_budget_s", "epochs_per_trial",
              "loss_threshold", "temperature"):
        assert p in n.inputs, f"missing input {p!r}"
    # `model` is NOT on the agent — the LLM node owns model selection.
    assert "model" not in n.inputs
    for p in ("control", "best_score", "best_hash", "history"):
        assert p in n.outputs, f"missing output {p!r}"
    # Initial state on a fresh agent.
    out = n.execute({k: n.inputs[k].default_value for k in n.inputs})
    assert out["control"] is None
    assert out["best_score"] == float("inf")
    assert out["best_hash"] == ""
    assert out["history"] == []


# ── collect_targets ────────────────────────────────────────────────────────

def test_collect_targets_walks_outgoing_control_wires():
    from plugins.agents._autoresearch.control_loop import collect_targets
    g, a, w, b, agent = _build_graph()
    targets = collect_targets(g, agent.id)
    target_ports = {(t.node_id, t.port_name) for t in targets}
    assert (w.id, "width") in target_ports
    assert (w.id, "activation") in target_ports
    assert (w.id, "rate") in target_ports
    assert len(targets) == 3
    # Each target carries port metadata for the LLM prompt.
    width_t = next(t for t in targets if t.port_name == "width")
    assert width_t.port_type == "INT"
    assert width_t.choices is None
    act_t = next(t for t in targets if t.port_name == "activation")
    assert act_t.port_type == "STRING"
    assert act_t.choices == ["relu", "gelu", "tanh"]


def test_collect_targets_skips_wires_into_missing_ports():
    """If a target node was deleted or its port renamed, the wire is
    silently skipped instead of crashing the loop."""
    from plugins.agents._autoresearch.control_loop import collect_targets
    g, a, w, b, agent = _build_graph()
    g.remove_node(w.id)   # leaves dangling control wires
    assert collect_targets(g, agent.id) == []


# ── parse_changes ──────────────────────────────────────────────────────────

def test_parse_changes_strips_fences_and_extracts_list():
    from plugins.agents._autoresearch.control_loop import parse_changes
    raw = """```json
{"changes": [{"target": "w.width", "value": 64},
              {"target": "w.activation", "value": "gelu"}]}
```"""
    out = parse_changes(raw)
    assert out == [{"target": "w.width", "value": 64},
                   {"target": "w.activation", "value": "gelu"}]


def test_parse_changes_accepts_empty_changes_list():
    from plugins.agents._autoresearch.control_loop import parse_changes
    assert parse_changes('{"changes": []}') == []


def test_parse_changes_rejects_bad_shapes():
    from plugins.agents._autoresearch.control_loop import parse_changes
    with pytest.raises(ValueError):
        parse_changes("not json at all")
    with pytest.raises(ValueError):
        parse_changes('{"foo": []}')
    with pytest.raises(ValueError):
        parse_changes('{"changes": [{"target": "x"}]}')   # missing value


# ── apply_changes ──────────────────────────────────────────────────────────

def test_apply_changes_writes_to_input_defaults():
    from plugins.agents._autoresearch.control_loop import (
        apply_changes, collect_targets,
    )
    g, a, w, b, agent = _build_graph()
    targets = collect_targets(g, agent.id)
    width_id = next(t.target_id for t in targets if t.port_name == "width")
    act_id   = next(t.target_id for t in targets if t.port_name == "activation")

    n_applied, errors = apply_changes(g, targets, [
        {"target": width_id, "value": 64},
        {"target": act_id,   "value": "gelu"},
    ])
    assert n_applied == 2
    assert errors == []
    assert g.nodes[w.id].inputs["width"].default_value == 64
    assert g.nodes[w.id].inputs["activation"].default_value == "gelu"


def test_apply_changes_rejects_invalid_choices():
    from plugins.agents._autoresearch.control_loop import (
        apply_changes, collect_targets,
    )
    g, a, w, b, agent = _build_graph()
    targets = collect_targets(g, agent.id)
    act_id = next(t.target_id for t in targets if t.port_name == "activation")

    n_applied, errors = apply_changes(g, targets, [
        {"target": act_id, "value": "swish"},   # not in choices
    ])
    assert n_applied == 0
    assert errors and "not in choices" in errors[0]
    # Original value preserved.
    assert g.nodes[w.id].inputs["activation"].default_value == "relu"


def test_apply_changes_coerces_int_target():
    from plugins.agents._autoresearch.control_loop import (
        apply_changes, collect_targets,
    )
    g, a, w, b, agent = _build_graph()
    targets = collect_targets(g, agent.id)
    width_id = next(t.target_id for t in targets if t.port_name == "width")

    # LLM emitted a stringified number — should be coerced to int.
    n_applied, errors = apply_changes(g, targets, [
        {"target": width_id, "value": "256"},
    ])
    assert n_applied == 1
    assert errors == []
    assert g.nodes[w.id].inputs["width"].default_value == 256


def test_apply_changes_skips_unknown_targets():
    from plugins.agents._autoresearch.control_loop import (
        apply_changes, collect_targets,
    )
    g, a, w, b, agent = _build_graph()
    targets = collect_targets(g, agent.id)
    n_applied, errors = apply_changes(g, targets, [
        {"target": "ghost.nope", "value": 1},
    ])
    assert n_applied == 0
    assert errors and "unknown target" in errors[0]


# ── ControlLoop end-to-end ─────────────────────────────────────────────────

def test_control_loop_runs_trials_and_keeps_best(tmp_path):
    """Three trials, scripted LLM proposals, scripted scores. The loop
    should record the trial with the lowest score as best."""
    from plugins.agents._autoresearch.ledger import Ledger
    from plugins.agents._autoresearch.control_loop import (
        ControlBudget, ControlLoop, collect_targets,
    )
    llm = _ScriptedLLM([])    # we'll patch in scripted responses below
    g, a, w, b, agent = _build_graph(llm)
    targets = collect_targets(g, agent.id)
    width_t = next(t for t in targets if t.port_name == "width")

    # Three proposed widths; trial 2 wins (score 0.3).
    llm._r = [
        json.dumps({"changes": [{"target": width_t.target_id, "value": 32}]}),
        json.dumps({"changes": [{"target": width_t.target_id, "value": 128}]}),
        json.dumps({"changes": [{"target": width_t.target_id, "value": 256}]}),
    ]
    fake = _FakeTrainOrch(scores=[0.7, 0.3, 0.5])
    reg = OrchestratorRegistry(g, [
        (["train_", "get_training_"], lambda graph: fake),
    ])

    ledger = Ledger(str(tmp_path / "results.tsv"))
    loop = ControlLoop(
        run_id="t1", graph=g, registry=reg, agent_node=agent,
        targets=targets, llm=llm, playbook="tune width",
        budget=ControlBudget(trials=3, wall_clock_s=10.0),
        ledger=ledger, metric="val_loss", eval_budget_s=1.0,
        train_start_params={"device": "cpu",
                            "datasets": {"task_1": {"path": "stub", "batch_size": 8}}},
    )
    loop.start()
    loop.join(timeout=5.0)

    assert loop.state.current_status == "done"
    assert loop.state.trials_done == 3
    assert loop.state.best_score == pytest.approx(0.3)
    # Trial 2 was kept, trial 3 was reverted → width is 128 (the winner).
    assert g.nodes[w.id].inputs["width"].default_value == 128
    # Agent state mirrored the loop's best score.
    assert agent._best_score == pytest.approx(0.3)
    assert len(agent._history) == 3


def test_control_loop_apply_rejected_records_crash_without_training(tmp_path):
    """If every change in a trial fails validation, the loop must NOT fire
    train_start — that would be wasted compute on an unmutated graph."""
    from plugins.agents._autoresearch.ledger import Ledger
    from plugins.agents._autoresearch.control_loop import (
        ControlBudget, ControlLoop, collect_targets,
    )
    llm = _ScriptedLLM([])
    g, a, w, b, agent = _build_graph(llm)
    targets = collect_targets(g, agent.id)
    act_t = next(t for t in targets if t.port_name == "activation")

    # Every proposal violates the choice constraint.
    llm._r = [json.dumps({"changes": [{"target": act_t.target_id,
                                          "value": "swish"}]})] * 3
    fake = _FakeTrainOrch(scores=[0.1, 0.1, 0.1])
    reg = OrchestratorRegistry(g, [
        (["train_", "get_training_"], lambda graph: fake),
    ])
    ledger = Ledger(str(tmp_path / "results.tsv"))
    loop = ControlLoop(
        run_id="t2", graph=g, registry=reg, agent_node=agent,
        targets=targets, llm=llm, playbook="x",
        budget=ControlBudget(trials=3, wall_clock_s=10.0),
        ledger=ledger,
    )
    loop.start()
    loop.join(timeout=5.0)
    assert loop.state.current_status == "done"
    # No train_start calls — every trial was rejected at apply time.
    assert not any(m == "train_start" for m, _ in fake.calls)
    # Every trial is recorded as a crash so the LLM sees its failures
    # in the next prompt's recent_results tail.
    assert all(h["status"] == "crash" for h in agent._history)


# ── autoresearch_start RPC validation ──────────────────────────────────────

def _make_orch_with_graph(g, fake_train=None):
    fake_train = fake_train or _FakeTrainOrch(scores=[0.5])
    reg = OrchestratorRegistry(g, [
        (["agent_", "get_agent_"], lambda graph: AgentOrchestrator(graph)),
        (["train_", "get_training_"], lambda graph: fake_train),
    ])
    return reg.resolve("agent_list_agent_nodes"), fake_train


def test_autoresearch_start_requires_agent_node():
    g = Graph()
    agent_orch, _ = _make_orch_with_graph(g)
    r = agent_orch.handle_rpc("agent_autoresearch_start", {})
    assert r["ok"] is False
    assert "AutoresearchAgent" in r["error"]


def test_autoresearch_start_requires_llm_wire():
    """Agent on canvas but no LLM connected → clear error."""
    g, a, w, b, agent = _build_graph(llm=None)
    agent_orch, _ = _make_orch_with_graph(g)
    r = agent_orch.handle_rpc("agent_autoresearch_start", {})
    assert r["ok"] is False
    assert "llm" in r["error"].lower()


def test_autoresearch_start_requires_control_wires():
    """Agent + LLM but no control wires → can't start (empty search space)."""
    from nodes.agents.autoresearch_agent import AutoresearchAgentNode

    llm = _ScriptedLLM(["{}"])
    g = Graph()
    agent = AutoresearchAgentNode(); g.add_node(agent)

    class _LLMNode(BaseNode):
        type_name = "_t_llm"
        def _setup_ports(self):
            self.add_output("llm", "LLM")
        def execute(self, inputs): return {"llm": llm}
    cli = _LLMNode(); g.add_node(cli)
    g.add_connection(cli.id, "llm", agent.id, "llm")
    # No control wires.

    agent_orch, _ = _make_orch_with_graph(g)
    r = agent_orch.handle_rpc("agent_autoresearch_start", {})
    assert r["ok"] is False
    assert "control" in r["error"].lower()


def test_autoresearch_starts_without_cached_training_params():
    """Zero manual training runs beforehand — autoresearch should still
    start, synthesizing `device` from hardware detection and letting
    A/B markers drive the rest. Removes the clunky 'run training once
    first' requirement."""
    llm = _ScriptedLLM([json.dumps({"changes": []})])
    g, a, w, b, agent = _build_graph(llm)
    agent.inputs["trials"].default_value = 1
    fake = _FakeTrainOrch(scores=[0.5], last_params=None)
    agent_orch, _ = _make_orch_with_graph(g, fake_train=fake)
    r = agent_orch.handle_rpc("agent_autoresearch_start", {})
    assert r["ok"] is True, r
    # Wait for completion so the test is deterministic.
    for _ in range(50):
        st = agent_orch.handle_rpc("agent_autoresearch_state",
                                     {"run_id": r["run_id"]})
        if st["current_status"] in ("done", "error", "stopped"):
            break
        time.sleep(0.02)
    # train_start fired with a sensible envelope.
    assert fake.calls
    starts = [p for m, p in fake.calls if m == "train_start"]
    assert starts
    assert starts[0]["device"] in ("cpu", "cuda:0")
    assert starts[0]["epochs"] >= 1


def test_autoresearch_start_kicks_off_run():
    llm = _ScriptedLLM([
        json.dumps({"changes": []}),
        json.dumps({"changes": []}),
    ])
    g, a, w, b, agent = _build_graph(llm)
    agent.inputs["trials"].default_value = 2
    agent_orch, _ = _make_orch_with_graph(g)
    r = agent_orch.handle_rpc("agent_autoresearch_start", {})
    assert r["ok"] is True, r
    run_id = r["run_id"]
    # Wait for completion.
    for _ in range(50):
        st = agent_orch.handle_rpc("agent_autoresearch_state", {"run_id": run_id})
        if st["current_status"] in ("done", "error", "stopped"):
            break
        time.sleep(0.05)
    assert st["current_status"] == "done"
    assert st["trials_done"] == 2
