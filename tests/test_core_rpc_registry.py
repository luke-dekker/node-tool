"""OrchestratorRegistry — method-prefix registry replacing the hardcoded
try/except dispatch chain. Plugins self-register via
`ctx.register_orchestrator(prefixes, factory)`.
"""
from __future__ import annotations
import pytest

from core.graph import Graph
from core.plugins import PluginContext, OrchestratorRegistry


class _FakeOrch:
    """Minimal stand-in for an orchestrator. Records every handle_rpc call."""

    def __init__(self, graph, owns: list[str]):
        self.graph = graph
        self.owns = list(owns)
        self.calls: list[tuple[str, dict]] = []

    def handle_rpc(self, method, params):
        self.calls.append((method, dict(params or {})))
        if not any(method.startswith(p) for p in self.owns):
            raise ValueError(f"Unknown method: {method}")
        return {"method": method, "params": params}


# ── register_orchestrator API ──────────────────────────────────────────────

def test_register_orchestrator_accepts_single_prefix():
    ctx = PluginContext()
    ctx.register_orchestrator("agent_", lambda g: _FakeOrch(g, ["agent_"]))
    factories = ctx.orchestrator_factories
    assert len(factories) == 1
    prefixes, factory = factories[0]
    assert prefixes == ["agent_"]


def test_register_orchestrator_accepts_prefix_list():
    ctx = PluginContext()
    ctx.register_orchestrator(
        ["train_", "get_training_", "drain_training_"],
        lambda g: _FakeOrch(g, ["train_"]),
    )
    prefixes, _ = ctx.orchestrator_factories[0]
    assert prefixes == ["train_", "get_training_", "drain_training_"]


# ── Resolution (prefix match) ──────────────────────────────────────────────

def test_resolve_picks_matching_prefix():
    g = Graph()
    reg = OrchestratorRegistry(g, [
        (["agent_"], lambda g: _FakeOrch(g, ["agent_"])),
        (["train_"], lambda g: _FakeOrch(g, ["train_"])),
    ])
    ag = reg.resolve("agent_start")
    tr = reg.resolve("train_start")
    assert ag is not tr
    assert ag.owns == ["agent_"]
    assert tr.owns == ["train_"]


def test_resolve_longest_prefix_wins():
    """get_agent_state matches both 'get_' (if registered) and 'get_agent_' —
    the longer prefix owner wins."""
    g = Graph()
    reg = OrchestratorRegistry(g, [
        (["get_"], lambda g: _FakeOrch(g, ["get_"])),            # generic
        (["get_agent_"], lambda g: _FakeOrch(g, ["get_agent_"])), # specific
    ])
    orch = reg.resolve("get_agent_state")
    assert orch.owns == ["get_agent_"]


def test_resolve_unknown_method_returns_none():
    g = Graph()
    reg = OrchestratorRegistry(g, [
        (["agent_"], lambda g: _FakeOrch(g, ["agent_"])),
    ])
    assert reg.resolve("nope_method") is None
    assert reg.resolve("") is None


def test_resolve_caches_instance_across_calls():
    g = Graph()
    built = []
    def _factory(gg):
        built.append(gg)
        return _FakeOrch(gg, ["agent_"])
    reg = OrchestratorRegistry(g, [(["agent_"], _factory)])
    o1 = reg.resolve("agent_start")
    o2 = reg.resolve("agent_stop")
    assert o1 is o2
    assert len(built) == 1   # factory called once


# ── Dispatch ───────────────────────────────────────────────────────────────

def test_try_dispatch_routes_to_owner():
    g = Graph()
    reg = OrchestratorRegistry(g, [
        (["agent_"], lambda g: _FakeOrch(g, ["agent_"])),
    ])
    result = reg.try_dispatch("agent_start", {"message": "hi"})
    assert result == {"method": "agent_start", "params": {"message": "hi"}}


def test_try_dispatch_unowned_returns_unhandled_sentinel():
    g = Graph()
    reg = OrchestratorRegistry(g, [])
    assert reg.try_dispatch("anything", {}) is OrchestratorRegistry._UNHANDLED


def test_try_dispatch_orchestrator_value_error_falls_through():
    """An orchestrator that owns 'agent_' but raises ValueError for a
    matched method is treated as unhandled (same semantics as the old
    try/except chain)."""
    g = Graph()

    class _Picky:
        def __init__(self, g): self.graph = g
        def handle_rpc(self, method, params):
            raise ValueError(f"nope: {method}")

    reg = OrchestratorRegistry(g, [(["agent_"], lambda g: _Picky(g))])
    assert reg.try_dispatch("agent_start", {}) is OrchestratorRegistry._UNHANDLED


# ── rebind_graph ───────────────────────────────────────────────────────────

def test_rebind_graph_updates_existing_orchestrators():
    g1 = Graph()
    reg = OrchestratorRegistry(g1, [
        (["agent_"], lambda g: _FakeOrch(g, ["agent_"])),
    ])
    orch = reg.resolve("agent_start")
    assert orch.graph is g1

    g2 = Graph()
    reg.rebind_graph(g2)
    assert reg.graph is g2
    assert orch.graph is g2   # same instance, new graph ref


def test_rebind_graph_before_first_resolve_is_safe():
    g1 = Graph()
    reg = OrchestratorRegistry(g1, [
        (["agent_"], lambda g: _FakeOrch(g, ["agent_"])),
    ])
    g2 = Graph()
    reg.rebind_graph(g2)   # no orchestrators built yet
    orch = reg.resolve("agent_start")
    assert orch.graph is g2


# ── End-to-end: PluginContext + registry ───────────────────────────────────

def test_plugin_context_round_trip():
    ctx = PluginContext()
    ctx.register_orchestrator(
        ["agent_", "get_agent_"],
        lambda g: _FakeOrch(g, ["agent_", "get_agent_"]),
    )
    ctx.register_orchestrator(
        ["train_"], lambda g: _FakeOrch(g, ["train_"]),
    )
    g = Graph()
    reg = OrchestratorRegistry(g, ctx.orchestrator_factories)
    assert reg.try_dispatch("agent_start", {})["method"] == "agent_start"
    assert reg.try_dispatch("get_agent_state", {})["method"] == "get_agent_state"
    assert reg.try_dispatch("train_start", {})["method"] == "train_start"
    assert reg.try_dispatch("unknown_method", {}) is OrchestratorRegistry._UNHANDLED
