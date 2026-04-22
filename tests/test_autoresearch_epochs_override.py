"""Regression test — autoresearch must override `epochs` with its own
per-trial budget, not inherit whatever the user set for full training.

Before this fix, every trial trained for as many epochs as the user last
submitted via the Training panel. Paired with a tight `eval_budget_s`,
every trial hit wall-clock timeout and was recorded as a crash — the
whole loop produced no usable information, just endless restarts.
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


class _StaticLLM:
    def __init__(self, response: str):
        self._r = response
    def chat(self, messages, *, model=None, temperature=0.5, **kw):
        from plugins.agents._llm.protocol import Message, ChatResult
        return ChatResult(message=Message(role="assistant", content=self._r))
    def stream(self, *a, **k):
        raise NotImplementedError


class _FakeTrainOrch:
    def __init__(self, cached_epochs=10):
        self.train_start_calls = []
        self._last_params = {
            "device": "cpu",
            "epochs": cached_epochs,  # simulate panel submission
            "datasets": {"task_1": {"path": "stub", "batch_size": 8}},
        }
    def handle_rpc(self, method, params):
        if method == "get_training_last_params":
            return {"params": self._last_params}
        if method == "train_start":
            self.train_start_calls.append(dict(params or {}))
            return {"ok": True}
        if method == "get_training_state":
            return {"status": "done"}
        if method == "get_training_losses":
            return {"series": {"train": [0.5], "val": [0.4]}}
        if method == "train_stop":
            return {"ok": True}
        raise ValueError(f"unknown: {method}")


class _Widget(BaseNode):
    type_name = "_t_w"
    def _setup_ports(self):
        self.add_input("width", PortType.INT, default=8)
        self.add_output("tensor_out", PortType.ANY)
    def execute(self, inputs): return {"tensor_out": None}


def _build():
    from nodes.agents.autoresearch_agent import AutoresearchAgentNode
    llm = _StaticLLM(json.dumps({"changes": []}))
    g = Graph()
    w = _Widget(); g.add_node(w)

    class _LLMNode(BaseNode):
        type_name = "_t_llm"
        def _setup_ports(self):
            self.add_output("llm", "LLM")
        def execute(self, inputs): return {"llm": llm}
    cli = _LLMNode(); g.add_node(cli)

    agent = AutoresearchAgentNode()
    agent.inputs["trials"].default_value = 1
    agent.inputs["eval_budget_s"].default_value = 5.0
    agent.inputs["wall_clock_s"].default_value = 10.0
    agent.inputs["epochs_per_trial"].default_value = 2
    g.add_node(agent)
    g.add_connection(cli.id, "llm", agent.id, "llm")
    g.add_connection(agent.id, "control", w.id, "width")
    return g, agent, w


def _wait(agent_orch, run_id, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = agent_orch.handle_rpc("agent_autoresearch_state", {"run_id": run_id})
        if st["current_status"] in ("done", "stopped", "error"):
            return st
        time.sleep(0.02)
    return agent_orch.handle_rpc("agent_autoresearch_state", {"run_id": run_id})


def test_autoresearch_overrides_epochs_with_agent_budget():
    """Agent's `epochs_per_trial` wins over the cached training params."""
    g, agent, w = _build()
    fake = _FakeTrainOrch(cached_epochs=10)   # user ran full training
    reg = OrchestratorRegistry(g, [
        (["agent_", "get_agent_"], lambda graph: AgentOrchestrator(graph)),
        (["train_", "get_training_"], lambda graph: fake),
    ])
    agent_orch = reg.resolve("agent_list_agent_nodes")

    r = agent_orch.handle_rpc("agent_autoresearch_start", {})
    assert r["ok"] is True, r
    _wait(agent_orch, r["run_id"])

    # train_start fired at least once; it received the SHORT per-trial
    # epochs budget, NOT the cached 10 from the panel.
    assert fake.train_start_calls
    call = fake.train_start_calls[0]
    assert call["epochs"] == 2
    # Other cached fields still flow through.
    assert call["datasets"] == {"task_1": {"path": "stub", "batch_size": 8}}
    assert call["group"] == "task_1"
