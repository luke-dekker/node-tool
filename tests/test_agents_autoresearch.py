"""Phase C autoresearch — textual serializer, mutation ops, ledger,
evaluator, experiment loop, and the three config nodes
(Mutator / Evaluator / ExperimentLoop).

Heavy-dep coupling is mocked out: `_FakeRegistry` stands in for the plugin
OrchestratorRegistry and `_FakeMutatorLLM` returns canned JSON ops —
nothing touches torch or the real training orchestrator.
"""
from __future__ import annotations
import json
import os
import threading
import time

import pytest

from core.graph import Graph
from core.node import BaseNode, MarkerRole, PortType


@pytest.fixture(autouse=True)
def _register_port_types():
    from plugins.agents.port_types import register_all
    register_all()


# ── Minimal marker stand-ins (no torch) ───────────────────────────────────

class _InputMarker(BaseNode):
    type_name = "_ar_input_marker"
    label = "A"
    marker_role = MarkerRole.INPUT
    def _setup_ports(self):
        self.add_input("group", PortType.STRING, default="task_1")
        self.add_output("tensor", PortType.ANY)
    def execute(self, inputs): return {"tensor": None}


class _TargetMarker(BaseNode):
    type_name = "_ar_target_marker"
    label = "B"
    marker_role = MarkerRole.TRAIN_TARGET
    def _setup_ports(self):
        self.add_input("tensor_in", PortType.ANY)
        self.add_input("group", PortType.STRING, default="task_1")
        self.add_output("config", PortType.ANY)
    def execute(self, inputs): return {"config": {}}


class _Widget(BaseNode):
    type_name = "_ar_widget"
    label = "Widget"
    def _setup_ports(self):
        self.add_input("x", PortType.ANY)
        self.add_input("scale", PortType.FLOAT, default=1.0)
        self.add_output("y", PortType.ANY)
    def execute(self, inputs):
        return {"y": inputs.get("x")}


class _BigWidget(BaseNode):
    type_name = "_ar_big_widget"
    label = "BigWidget"
    def _setup_ports(self):
        self.add_input("x", PortType.ANY)
        self.add_input("scale", PortType.FLOAT, default=1.0)
        self.add_output("y", PortType.ANY)
    def execute(self, inputs): return {"y": inputs.get("x")}


@pytest.fixture(autouse=True)
def _register_test_nodes():
    from nodes import NODE_REGISTRY
    for cls in (_InputMarker, _TargetMarker, _Widget, _BigWidget):
        NODE_REGISTRY[cls.type_name] = cls
    yield
    for cls in (_InputMarker, _TargetMarker, _Widget, _BigWidget):
        NODE_REGISTRY.pop(cls.type_name, None)


def _build_ab_graph(group="task_1"):
    g = Graph()
    a = _InputMarker()
    w = _Widget()
    b = _TargetMarker()
    for n in (a, w, b):
        g.add_node(n)
    a.inputs["group"].default_value = group
    b.inputs["group"].default_value = group
    g.add_connection(a.id, "tensor", w.id, "x")
    g.add_connection(w.id, "y", b.id, "tensor_in")
    return g, a, w, b


# ══════════════════════════════════════════════════════════════════════════
#  Textual serializer
# ══════════════════════════════════════════════════════════════════════════

def test_textual_contains_class_names_and_short_ids():
    from plugins.agents._autoresearch.graph_textual import serialize_graph_textual
    g, a, w, b = _build_ab_graph()
    text = serialize_graph_textual(g)
    assert "_InputMarker" in text
    assert "_Widget" in text
    assert "_TargetMarker" in text
    assert a.id[:6] in text
    assert w.id[:6] in text


def test_textual_inputs_render_connections_and_defaults():
    from plugins.agents._autoresearch.graph_textual import serialize_graph_textual
    g, a, w, b = _build_ab_graph()
    text = serialize_graph_textual(g)
    # The widget's x input is connected to A.tensor — must render as <... .tensor>
    assert f".tensor>" in text
    # Scalars render via repr
    assert "scale=1.0" in text


def test_textual_region_filter_excludes_outside_nodes():
    from plugins.agents._autoresearch.graph_textual import serialize_graph_textual
    g, a, w, b = _build_ab_graph()
    outside = _Widget()
    g.add_node(outside)
    text = serialize_graph_textual(g, region=[a.id, w.id, b.id])
    assert outside.id[:6] not in text


def test_textual_truncates_to_max_chars():
    from plugins.agents._autoresearch.graph_textual import serialize_graph_textual
    g = Graph()
    for _ in range(50):
        g.add_node(_Widget())
    short = serialize_graph_textual(g, max_chars=200)
    assert "omitted" in short or len(short) <= 400


# ══════════════════════════════════════════════════════════════════════════
#  Mutation ops + parse + apply
# ══════════════════════════════════════════════════════════════════════════

def test_parse_mutation_json_strips_code_fences():
    from plugins.agents._autoresearch.mutation import parse_mutation_json
    raw = "```json\n" \
          '{"op": "set_input", "node_id": "abcdef", "port": "scale", "value": 2.5}' \
          "\n```"
    op = parse_mutation_json(raw)
    assert op.op == "set_input"
    assert op.port == "scale"
    assert op.value == 2.5


def test_parse_mutation_json_unwraps_envelope():
    from plugins.agents._autoresearch.mutation import parse_mutation_json
    raw = '{"mutation": {"op": "remove_node", "node_id": "abcdef"}}'
    op = parse_mutation_json(raw)
    assert op.op == "remove_node"


def test_parse_mutation_json_rejects_unknown_op():
    from plugins.agents._autoresearch.mutation import parse_mutation_json
    with pytest.raises(ValueError, match="Unknown op"):
        parse_mutation_json('{"op": "delete_everything"}')


def test_apply_swap_node_class():
    from plugins.agents._autoresearch.mutation import MutationOp, apply_mutation
    g, a, w, b = _build_ab_graph()
    op = MutationOp(op="swap_node_class", node_id=w.id, new_class_name="_ar_big_widget")
    ok, msg = apply_mutation(g, op)
    assert ok, msg
    assert type(g.nodes[w.id]).__name__ == "_BigWidget"
    # Defaults copied over
    assert g.nodes[w.id].inputs["scale"].default_value == 1.0


def test_apply_swap_respects_allowlist():
    from plugins.agents._autoresearch.mutation import MutationOp, apply_mutation
    g, a, w, b = _build_ab_graph()
    op = MutationOp(op="swap_node_class", node_id=w.id, new_class_name="_ar_big_widget")
    ok, msg = apply_mutation(g, op, allowlist={"_ar_widget"})
    assert not ok and "allowlist" in msg


def test_apply_swap_respects_cone():
    from plugins.agents._autoresearch.mutation import MutationOp, apply_mutation
    g, a, w, b = _build_ab_graph()
    op = MutationOp(op="swap_node_class", node_id=w.id, new_class_name="_ar_big_widget")
    ok, msg = apply_mutation(g, op, cone={a.id, b.id})   # w not in cone
    assert not ok and "cone" in msg


def test_apply_set_input():
    from plugins.agents._autoresearch.mutation import MutationOp, apply_mutation
    g, a, w, b = _build_ab_graph()
    op = MutationOp(op="set_input", node_id=w.id, port="scale", value=4.2)
    ok, _ = apply_mutation(g, op)
    assert ok
    assert g.nodes[w.id].inputs["scale"].default_value == 4.2


def test_apply_set_input_rejects_path_escape():
    from plugins.agents._autoresearch.mutation import MutationOp, apply_mutation
    g, a, w, b = _build_ab_graph()
    op = MutationOp(op="set_input", node_id=w.id, port="scale",
                    value="../../etc/passwd")
    ok, msg = apply_mutation(g, op)
    assert not ok and "escapes cwd" in msg


def test_apply_remove_node():
    from plugins.agents._autoresearch.mutation import MutationOp, apply_mutation
    g, a, w, b = _build_ab_graph()
    op = MutationOp(op="remove_node", node_id=w.id)
    ok, _ = apply_mutation(g, op)
    assert ok
    assert w.id not in g.nodes


def test_apply_add_node():
    from plugins.agents._autoresearch.mutation import MutationOp, apply_mutation
    g, a, w, b = _build_ab_graph()
    op = MutationOp(op="add_node", class_name="_ar_widget",
                    connections=[[w.id, "y", "x"]])
    ok, _ = apply_mutation(g, op)
    assert ok
    # New node exists + is connected as specified
    new_ids = set(g.nodes.keys()) - {a.id, w.id, b.id}
    assert len(new_ids) == 1
    new_id = next(iter(new_ids))
    assert any(c.from_node_id == w.id and c.to_node_id == new_id
               for c in g.connections)


def test_apply_add_connection_rejects_cycle():
    from plugins.agents._autoresearch.mutation import MutationOp, apply_mutation
    g, a, w, b = _build_ab_graph()
    # Adding b → a would cycle
    op = MutationOp(op="add_connection",
                    from_id=b.id, from_port="config",
                    to_id=a.id, to_port="group")
    ok, msg = apply_mutation(g, op)
    assert not ok


# ══════════════════════════════════════════════════════════════════════════
#  Ledger
# ══════════════════════════════════════════════════════════════════════════

def test_ledger_creates_file_with_header(tmp_path):
    from plugins.agents._autoresearch.ledger import Ledger, LEDGER_COLUMNS
    path = str(tmp_path / "run_x" / "results.tsv")
    Ledger(path)
    content = open(path, encoding="utf-8").read()
    assert content.strip() == "\t".join(LEDGER_COLUMNS)


def test_ledger_append_writes_row(tmp_path):
    from plugins.agents._autoresearch.ledger import Ledger, LedgerRow
    path = str(tmp_path / "run_y" / "results.tsv")
    l = Ledger(path)
    l.append(LedgerRow(trial_idx=1, graph_hash="abc123",
                       op_kind="set_input", score=0.42,
                       status="keep", wall_clock_s=12.5))
    content = open(path, encoding="utf-8").read().splitlines()
    assert len(content) == 2
    fields = content[1].split("\t")
    assert fields[0] == "1"
    assert fields[3] == "0.420000"


def test_ledger_tail_returns_last_n(tmp_path):
    from plugins.agents._autoresearch.ledger import Ledger, LedgerRow
    l = Ledger(str(tmp_path / "tail" / "results.tsv"))
    for i in range(5):
        l.append(LedgerRow(trial_idx=i + 1, graph_hash=f"h{i}",
                           op_kind="k", score=float(i), status="keep",
                           wall_clock_s=1.0))
    tail = l.tail(n=2)
    assert tail.count("\n") == 2  # header + 2 rows
    assert "h3" in tail and "h4" in tail
    assert "h0" not in tail


# ══════════════════════════════════════════════════════════════════════════
#  run_eval (against fake registry)
# ══════════════════════════════════════════════════════════════════════════

class _FakeRegistry:
    """Minimal stand-in for OrchestratorRegistry with canned responses."""
    _UNHANDLED = object()

    def __init__(self, state_sequence, losses, *, raise_on=None):
        self.state_sequence = list(state_sequence)
        self.losses = losses
        self.raise_on = raise_on or ()
        self.calls: list[tuple[str, dict]] = []

    def try_dispatch(self, method, params):
        self.calls.append((method, dict(params or {})))
        if method in self.raise_on:
            raise RuntimeError(f"injected failure on {method}")
        if method == "train_start":
            return {"ok": True}
        if method == "get_training_state":
            if self.state_sequence:
                return self.state_sequence.pop(0)
            return {"status": "done"}
        if method == "get_training_losses":
            return self.losses
        if method == "train_stop":
            return {"ok": True}
        return self._UNHANDLED


def test_run_eval_scores_final_val_loss():
    from plugins.agents._autoresearch.evaluator import run_eval
    reg = _FakeRegistry(
        state_sequence=[{"status": "running"}, {"status": "done"}],
        losses={"val_loss": [0.9, 0.5, 0.42]},
    )
    r = run_eval(registry=reg, metric="val_loss", budget_s=5.0, poll_ms=5)
    assert r.score == pytest.approx(0.42)
    assert r.status == "keep"   # no best_so_far given → defaults to inf


def test_run_eval_returns_discard_when_score_worse_than_best():
    from plugins.agents._autoresearch.evaluator import run_eval
    reg = _FakeRegistry(
        state_sequence=[{"status": "done"}],
        losses={"val_loss": [0.99]},
    )
    r = run_eval(registry=reg, metric="val_loss", budget_s=5.0,
                 poll_ms=5, best_so_far=0.5)
    assert r.status == "discard"


def test_run_eval_returns_crash_on_error_status():
    from plugins.agents._autoresearch.evaluator import run_eval
    reg = _FakeRegistry(
        state_sequence=[{"status": "error", "error": "OOM"}],
        losses={},
    )
    r = run_eval(registry=reg, metric="val_loss", budget_s=5.0, poll_ms=5)
    assert r.status == "crash"
    assert "OOM" in r.error


def test_run_eval_respects_stop_flag():
    from plugins.agents._autoresearch.evaluator import run_eval
    reg = _FakeRegistry(
        state_sequence=[{"status": "running"}] * 50,
        losses={"val_loss": [0.1]},
    )
    flag = threading.Event()
    flag.set()
    r = run_eval(registry=reg, metric="val_loss", budget_s=5.0,
                 poll_ms=5, stop_flag=flag)
    assert r.status == "crash"
    assert "abort" in r.error.lower()


def test_run_eval_respects_wall_clock_budget():
    from plugins.agents._autoresearch.evaluator import run_eval
    # Always "running" — force the budget to trip
    reg = _FakeRegistry(
        state_sequence=[{"status": "running"}] * 1000,
        losses={"val_loss": [0.5]},
    )
    r = run_eval(registry=reg, metric="val_loss", budget_s=0.05,
                 poll_ms=10)
    assert r.status == "crash"
    assert "timeout" in r.error.lower()


def test_run_eval_returns_crash_if_no_training_orch():
    from plugins.agents._autoresearch.evaluator import run_eval
    class _EmptyRegistry:
        _UNHANDLED = object()
        def try_dispatch(self, method, params):
            return self._UNHANDLED
    r = run_eval(registry=_EmptyRegistry(), metric="val_loss",
                 budget_s=1.0, poll_ms=5)
    assert r.status == "crash"
    assert "no training orchestrator" in r.error


# ══════════════════════════════════════════════════════════════════════════
#  ExperimentLoop
# ══════════════════════════════════════════════════════════════════════════

def test_experiment_loop_runs_budget_trials(tmp_path):
    """Loop runs exactly `trials` cycles with a fake mutator + registry."""
    from plugins.agents._autoresearch.ledger import Ledger
    from plugins.agents._autoresearch.loop import ExperimentLoop, LoopBudget

    g, a, w, b = _build_ab_graph()
    ledger = Ledger(str(tmp_path / "run" / "results.tsv"))
    # Three canned mutations: set scale to 2, 3, then swap class
    mutations = iter([
        json.dumps({"op": "set_input", "node_id": w.id, "port": "scale", "value": 2.0}),
        json.dumps({"op": "set_input", "node_id": w.id, "port": "scale", "value": 3.0}),
        json.dumps({"op": "swap_node_class", "node_id": w.id,
                    "new_class_name": "_ar_big_widget"}),
    ])
    def _mutator(_tail): return next(mutations)

    # Descending scores so every trial is a "keep"
    scores = iter([0.9, 0.5, 0.2])
    class _SeqRegistry:
        _UNHANDLED = object()
        def __init__(self): self.calls = 0
        def try_dispatch(self, m, p):
            if m == "train_start":  return {"ok": True}
            if m == "get_training_state": return {"status": "done"}
            if m == "get_training_losses":
                return {"val_loss": [next(scores)]}
            return self._UNHANDLED

    loop = ExperimentLoop(
        run_id="test-run", graph=g, registry=_SeqRegistry(),
        mutator_fn=_mutator,
        budget=LoopBudget(trials=3, wall_clock_s=10.0),
        ledger=ledger, eval_budget_s=5.0,
    )
    loop.start()
    loop.join(timeout=5.0)
    assert loop.state.trials_done == 3
    assert loop.state.best_score == pytest.approx(0.2)
    assert loop.state.current_status == "done"
    assert len(loop.state.history) == 3


def test_experiment_loop_reverts_on_discard(tmp_path):
    """Trial 1 establishes a baseline (always kept, no history). Trial 2 is
    worse — that mutation must be reverted."""
    from plugins.agents._autoresearch.ledger import Ledger
    from plugins.agents._autoresearch.loop import ExperimentLoop, LoopBudget

    g, a, w, b = _build_ab_graph()
    w.inputs["scale"].default_value = 1.0
    ledger = Ledger(str(tmp_path / "rev" / "results.tsv"))

    muts = iter([
        json.dumps({"op": "set_input", "node_id": w.id, "port": "scale", "value": 5.0}),
        json.dumps({"op": "set_input", "node_id": w.id, "port": "scale", "value": 99.0}),
    ])
    def _mutator(_tail): return next(muts)

    scores = iter([0.5, 5.0])   # trial 2 is worse → discard
    class _Reg:
        _UNHANDLED = object()
        def try_dispatch(self, m, p):
            if m == "train_start": return {"ok": True}
            if m == "get_training_state": return {"status": "done"}
            if m == "get_training_losses":
                return {"val_loss": [next(scores)]}
            return self._UNHANDLED

    loop = ExperimentLoop(
        run_id="rev-run", graph=g, registry=_Reg(),
        mutator_fn=_mutator,
        budget=LoopBudget(trials=2, wall_clock_s=10.0),
        ledger=ledger,
    )
    loop.start()
    loop.join(timeout=5.0)
    # Trial 1 kept (scale=5.0, score=0.5); trial 2 reverted back to 5.0
    assert g.nodes[w.id].inputs["scale"].default_value == 5.0
    assert loop.state.best_score == pytest.approx(0.5)
    assert loop.state.history[0]["status"] == "keep"
    assert loop.state.history[1]["status"] == "discard"


def test_experiment_loop_parse_error_is_crash_not_raise(tmp_path):
    from plugins.agents._autoresearch.ledger import Ledger
    from plugins.agents._autoresearch.loop import ExperimentLoop, LoopBudget
    g, a, w, b = _build_ab_graph()
    ledger = Ledger(str(tmp_path / "bad" / "results.tsv"))

    def _mutator(_tail):
        return "not json at all"

    class _Reg:
        _UNHANDLED = object()
        def try_dispatch(self, *a, **kw): return {"ok": True}

    loop = ExperimentLoop(
        run_id="bad-run", graph=g, registry=_Reg(), mutator_fn=_mutator,
        budget=LoopBudget(trials=1, wall_clock_s=5.0), ledger=ledger,
    )
    loop.start()
    loop.join(timeout=5.0)
    row = loop.state.history[0]
    assert row["status"] == "crash"
    assert row["op_kind"] == "parse_error"


def test_experiment_loop_stop_mid_flight(tmp_path):
    from plugins.agents._autoresearch.ledger import Ledger
    from plugins.agents._autoresearch.loop import ExperimentLoop, LoopBudget
    g, a, w, b = _build_ab_graph()
    ledger = Ledger(str(tmp_path / "stop" / "results.tsv"))

    gate = threading.Event()
    def _mutator(_tail):
        gate.set()
        time.sleep(0.1)
        return json.dumps({"op": "set_input", "node_id": w.id,
                           "port": "scale", "value": 2.0})

    class _Reg:
        _UNHANDLED = object()
        def try_dispatch(self, m, p):
            if m == "train_start": return {"ok": True}
            if m == "get_training_state": return {"status": "done"}
            if m == "get_training_losses": return {"val_loss": [0.1]}
            return self._UNHANDLED

    loop = ExperimentLoop(
        run_id="stop-run", graph=g, registry=_Reg(), mutator_fn=_mutator,
        budget=LoopBudget(trials=100, wall_clock_s=30.0), ledger=ledger,
    )
    loop.start()
    gate.wait(timeout=2.0)
    loop.stop()
    loop.join(timeout=5.0)
    assert loop.state.current_status in ("stopped", "done")
    assert loop.state.trials_done < 100


# ══════════════════════════════════════════════════════════════════════════
#  Nodes (config emitters)
# ══════════════════════════════════════════════════════════════════════════

def test_evaluator_node_emits_config():
    from nodes.agents.evaluator import EvaluatorNode
    out = EvaluatorNode().execute({
        "metric": "val_loss", "budget_seconds": 30.0,
        "epochs": 3, "group": "task_1",
    })
    spec = out["eval_spec"]
    assert spec == {"metric": "val_loss", "budget_seconds": 30.0,
                    "epochs": 3, "group": "task_1"}


def test_experiment_loop_node_emits_config():
    from nodes.agents.experiment_loop import ExperimentLoopNode
    out = ExperimentLoopNode().execute({
        "trials": 10, "wall_clock_s": 120.0,
        "loss_threshold": 0.1, "allowlist": "LinearNode, ReLUNode",
    })
    spec = out["loop_spec"]
    assert spec["trials"] == 10
    assert spec["wall_clock_s"] == 120.0
    assert spec["loss_threshold"] == 0.1
    assert spec["allowlist"] == ["LinearNode", "ReLUNode"]


def test_experiment_loop_node_zero_threshold_becomes_none():
    from nodes.agents.experiment_loop import ExperimentLoopNode
    out = ExperimentLoopNode().execute({
        "trials": 1, "wall_clock_s": 1.0,
        "loss_threshold": 0.0, "allowlist": "",
    })
    assert out["loop_spec"]["loss_threshold"] is None


# ══════════════════════════════════════════════════════════════════════════
#  MutatorNode — drives LLM, emits op dict
# ══════════════════════════════════════════════════════════════════════════

class _FakeMutatorLLM:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []
    def chat(self, messages, *, model=None, temperature=0.5, **kw):
        from plugins.agents._llm.protocol import Message, ChatResult
        self.calls.append({"messages": messages,
                            "model": model, "temperature": temperature})
        return ChatResult(
            message=Message(role="assistant", content=self.response),
            model=model or "fake",
        )


def test_mutator_node_emits_op_from_llm_json_response():
    from nodes.agents.mutator import MutatorNode
    g, a, w, b = _build_ab_graph()
    mut = MutatorNode()
    g.add_node(mut)
    llm = _FakeMutatorLLM(
        '{"op": "set_input", "node_id": "' + w.id +
        '", "port": "scale", "value": 2.0}'
    )
    stored, _, errors = g.execute()
    # Seed LLM default on the MutatorNode and rerun. Actually we'll pass
    # inputs directly via execute() since connected LLM values flow in on
    # graph execute — but nothing is connected here. Use node.execute directly.
    result = mut.execute({
        "llm": llm, "group": "task_1",
        "playbook": "tweak scale", "recent_results": "",
        "model": "", "temperature": 0.0,
    })
    assert result["mutation"]["op"] == "set_input"
    assert result["mutation"]["node_id"] == w.id
    assert result["mutation"]["value"] == 2.0
    # Prompt includes the textual graph
    assert "_Widget" in result["prompt"]


def test_mutator_node_requires_graph():
    from nodes.agents.mutator import MutatorNode
    mut = MutatorNode()
    with pytest.raises(RuntimeError, match="live Graph"):
        mut.execute({
            "llm": _FakeMutatorLLM("{}"), "group": "task_1",
            "playbook": "", "recent_results": "",
            "model": "", "temperature": 0.0,
        })


def test_mutator_node_surfaces_invalid_json():
    from nodes.agents.mutator import MutatorNode
    g, a, w, b = _build_ab_graph()
    mut = MutatorNode()
    g.add_node(mut)
    # Prime _graph via graph.execute (it runs the mutator; we don't read result)
    llm = _FakeMutatorLLM("not json at all")
    # Trigger the _graph attr by running graph.execute (errors will be captured
    # but we just want _graph bound).
    for _ in g.execute()[0]:
        pass
    with pytest.raises(RuntimeError, match="parseable"):
        mut.execute({
            "llm": llm, "group": "task_1", "playbook": "", "recent_results": "",
            "model": "", "temperature": 0.0,
        })


# ══════════════════════════════════════════════════════════════════════════
#  Plugin registers all three Phase C nodes
# ══════════════════════════════════════════════════════════════════════════

def test_phase_c_nodes_registered():
    from core.plugins import PluginContext
    import plugins.agents as agents_pkg
    ctx = PluginContext()
    agents_pkg.register(ctx)
    names = {c.type_name for c in ctx.node_classes}
    for t in ("ag_mutator", "ag_evaluator", "ag_experiment_loop"):
        assert t in names
