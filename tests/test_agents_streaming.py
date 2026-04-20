"""Phase B streaming — agent_start_stream, drain_tokens, cooperative stop.

The orchestrator spawns a background thread that pulls pieces from
`LLMClient.stream()` and deposits them in a per-session buffer.
`agent_drain_tokens` pops NEW chunks since the last drain and reports
a `done` flag once the thread has finished.

Tests use a StreamLLM that accepts an externally-controlled cadence
(Event-gated) so we can exercise partial-drain semantics deterministically
without relying on timing.
"""
from __future__ import annotations
import threading
import time
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _register_agent_port_types():
    from plugins.agents.port_types import register_all
    register_all()


# ── Stream-controllable mock LLM ───────────────────────────────────────────

class GatedStreamLLM:
    """Yields one piece per `release()` call. Lets tests step the stream
    deterministically — no sleep-based flakiness.

    `fail_at` (optional int): raise RuntimeError after that many pieces.
    """
    capabilities = {"stream"}

    def __init__(self, pieces, *, fail_at=None):
        self._pieces = list(pieces)
        self._gate = threading.Event()
        self._consumed = 0
        self._fail_at = fail_at
        self.calls: list[dict] = []

    def release(self, n: int = 1) -> None:
        """Allow the next N pieces to be yielded."""
        self._consumed_target = self._consumed + n
        self._gate.set()

    def release_all(self) -> None:
        self._consumed_target = len(self._pieces) + 1
        self._gate.set()

    def chat(self, messages, **kw):
        raise NotImplementedError("Tests should use .stream() path")

    def stream(self, messages, **kw) -> Iterator[str]:
        self.calls.append({"messages": list(messages), "kw": dict(kw)})
        self._consumed_target = 0
        for idx, piece in enumerate(self._pieces):
            # Block until the test releases this piece
            while self._consumed >= self._consumed_target:
                if not self._gate.wait(timeout=2.0):
                    raise RuntimeError("GatedStreamLLM: release() never called")
                self._gate.clear()
            self._consumed += 1
            if self._fail_at is not None and idx + 1 == self._fail_at:
                raise RuntimeError(f"injected failure at piece {idx + 1}")
            yield piece


class PlainStreamLLM:
    """Yields all pieces up front — for tests that just want a full transcript."""
    capabilities = {"stream"}

    def __init__(self, pieces):
        self._pieces = list(pieces)
        self.calls: list[dict] = []

    def chat(self, messages, **kw):
        raise NotImplementedError

    def stream(self, messages, **kw) -> Iterator[str]:
        self.calls.append({"messages": list(messages), "kw": dict(kw)})
        for p in self._pieces:
            yield p


# ── Shared graph-building helper ──────────────────────────────────────────

def _build_orch_with_llm(monkeypatch, llm):
    from core.graph import Graph
    from nodes.agents.agent import AgentNode
    from plugins.agents.agents_orchestrator import AgentOrchestrator

    g = Graph()
    ag = AgentNode()
    g.nodes[ag.id] = ag
    orch = AgentOrchestrator(g)
    monkeypatch.setattr(orch, "_resolve_llm_for_agent", lambda node: llm)
    return orch, ag


def _wait_for_done(orch, sid: str, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if orch._stream_done.get(sid):
            return
        time.sleep(0.01)
    raise AssertionError(f"Stream {sid[:6]} did not finish within {timeout}s")


# ── Tests ──────────────────────────────────────────────────────────────────

def test_start_stream_returns_session_id(monkeypatch):
    llm = PlainStreamLLM(["hello"])
    orch, ag = _build_orch_with_llm(monkeypatch, llm)
    resp = orch.handle_rpc("agent_start_stream",
                            {"agent_id": ag.id, "message": "hi"})
    assert resp["ok"] is True
    assert resp["session_id"]
    _wait_for_done(orch, resp["session_id"])


def test_start_stream_rejects_empty_message(monkeypatch):
    orch, ag = _build_orch_with_llm(monkeypatch, PlainStreamLLM(["x"]))
    resp = orch.handle_rpc("agent_start_stream",
                            {"agent_id": ag.id, "message": ""})
    assert resp["ok"] is False
    assert "Empty message" in resp["error"]


def test_start_stream_requires_agent_node(monkeypatch):
    from core.graph import Graph
    from plugins.agents.agents_orchestrator import AgentOrchestrator
    orch = AgentOrchestrator(Graph())
    resp = orch.handle_rpc("agent_start_stream", {"message": "hi"})
    assert resp["ok"] is False


def test_start_stream_requires_llm(monkeypatch):
    from core.graph import Graph
    from nodes.agents.agent import AgentNode
    from plugins.agents.agents_orchestrator import AgentOrchestrator
    g = Graph()
    ag = AgentNode()
    g.nodes[ag.id] = ag
    orch = AgentOrchestrator(g)
    # No resolver patched → LLM is None
    resp = orch.handle_rpc("agent_start_stream",
                            {"agent_id": ag.id, "message": "hi"})
    assert resp["ok"] is False
    assert "No LLM" in resp["error"]


def test_start_stream_rejects_backend_without_stream(monkeypatch):
    from core.graph import Graph
    from nodes.agents.agent import AgentNode
    from plugins.agents.agents_orchestrator import AgentOrchestrator

    class NoStreamBackend:
        def chat(self, *a, **kw): pass

    g = Graph()
    ag = AgentNode()
    g.nodes[ag.id] = ag
    orch = AgentOrchestrator(g)
    monkeypatch.setattr(orch, "_resolve_llm_for_agent",
                         lambda node: NoStreamBackend())
    resp = orch.handle_rpc("agent_start_stream",
                            {"agent_id": ag.id, "message": "hi"})
    assert resp["ok"] is False
    assert "does not support streaming" in resp["error"]


def test_drain_tokens_incremental_delivery(monkeypatch):
    """Gated stream — drain sees pieces as they're released, never duplicates."""
    llm = GatedStreamLLM(["Hello", " ", "world", "!"])
    orch, ag = _build_orch_with_llm(monkeypatch, llm)
    resp = orch.handle_rpc("agent_start_stream",
                            {"agent_id": ag.id, "message": "go"})
    sid = resp["session_id"]

    # Release the first two pieces
    llm.release(2)
    _poll_until(orch, sid, min_total=2)
    d1 = orch.handle_rpc("agent_drain_tokens", {"session_id": sid})
    assert d1["chunks"] == ["Hello", " "]
    assert d1["done"] is False

    # Release the rest
    llm.release_all()
    _wait_for_done(orch, sid)

    d2 = orch.handle_rpc("agent_drain_tokens", {"session_id": sid})
    assert d2["chunks"] == ["world", "!"]
    assert d2["done"] is True

    # Subsequent drains are empty but still report done
    d3 = orch.handle_rpc("agent_drain_tokens", {"session_id": sid})
    assert d3["chunks"] == []
    assert d3["done"] is True


def test_drain_without_session_id_targets_most_recent(monkeypatch):
    llm = PlainStreamLLM(["a", "b", "c"])
    orch, ag = _build_orch_with_llm(monkeypatch, llm)
    resp = orch.handle_rpc("agent_start_stream",
                            {"agent_id": ag.id, "message": "go"})
    _wait_for_done(orch, resp["session_id"])
    d = orch.handle_rpc("agent_drain_tokens", {})  # no session_id
    assert d["chunks"] == ["a", "b", "c"]
    assert d["done"] is True


def test_drain_with_no_active_sessions_returns_done():
    from core.graph import Graph
    from plugins.agents.agents_orchestrator import AgentOrchestrator
    orch = AgentOrchestrator(Graph())
    d = orch.handle_rpc("agent_drain_tokens", {})
    assert d["chunks"] == []
    assert d["done"] is True


def test_stop_cancels_stream_mid_flight(monkeypatch):
    llm = GatedStreamLLM(["piece1", "piece2", "piece3", "piece4", "piece5"])
    orch, ag = _build_orch_with_llm(monkeypatch, llm)
    resp = orch.handle_rpc("agent_start_stream",
                            {"agent_id": ag.id, "message": "go"})
    sid = resp["session_id"]

    llm.release(1)
    _poll_until(orch, sid, min_total=1)

    orch.handle_rpc("agent_stop", {"session_id": sid})
    # Let the stream advance past the stop flag check
    llm.release_all()
    _wait_for_done(orch, sid)

    state = orch.handle_rpc("get_agent_state", {"session_id": sid})
    assert state["status"] == "Aborted"
    # The reply must reflect what was emitted BEFORE the abort — not everything.
    assert state["reply"].startswith("piece1")
    assert "piece5" not in state["reply"]


def test_stream_error_surfaces_on_state(monkeypatch):
    llm = GatedStreamLLM(["alpha", "beta", "gamma"], fail_at=2)
    orch, ag = _build_orch_with_llm(monkeypatch, llm)
    resp = orch.handle_rpc("agent_start_stream",
                            {"agent_id": ag.id, "message": "go"})
    sid = resp["session_id"]
    llm.release_all()
    _wait_for_done(orch, sid)

    state = orch.handle_rpc("get_agent_state", {"session_id": sid})
    assert state["status"] == "Error"
    assert "injected failure" in state["error"]
    # Reply holds whatever pieces arrived before the raise
    assert state["reply"] == "alpha"


def test_state_reflects_streaming_then_done(monkeypatch):
    llm = GatedStreamLLM(["one", "two"])
    orch, ag = _build_orch_with_llm(monkeypatch, llm)
    resp = orch.handle_rpc("agent_start_stream",
                            {"agent_id": ag.id, "message": "go"})
    sid = resp["session_id"]
    # While the stream is gated, status is "Streaming"
    state = orch.handle_rpc("get_agent_state", {"session_id": sid})
    assert state["status"] == "Streaming"
    llm.release_all()
    _wait_for_done(orch, sid)
    state = orch.handle_rpc("get_agent_state", {"session_id": sid})
    assert state["status"] == "Done"
    assert state["reply"] == "onetwo"
    assert state["tokens_out"] == 2


def test_handle_rpc_routes_start_stream(monkeypatch):
    from core.graph import Graph
    from plugins.agents.agents_orchestrator import AgentOrchestrator
    orch = AgentOrchestrator(Graph())
    # No agent present → ok:False, but the method must route (not raise)
    resp = orch.handle_rpc("agent_start_stream", {"message": "x"})
    assert resp["ok"] is False


def test_two_concurrent_streams_dont_cross_contaminate(monkeypatch):
    """Two sessions on two AgentNodes drain independently."""
    from core.graph import Graph
    from nodes.agents.agent import AgentNode
    from plugins.agents.agents_orchestrator import AgentOrchestrator

    g = Graph()
    a1, a2 = AgentNode(), AgentNode()
    g.nodes[a1.id] = a1
    g.nodes[a2.id] = a2

    llms = {a1.id: PlainStreamLLM(["a1-x", "a1-y"]),
            a2.id: PlainStreamLLM(["a2-x", "a2-y", "a2-z"])}

    orch = AgentOrchestrator(g)
    monkeypatch.setattr(orch, "_resolve_llm_for_agent",
                         lambda node: llms[node.id])

    r1 = orch.handle_rpc("agent_start_stream",
                          {"agent_id": a1.id, "message": "go"})
    r2 = orch.handle_rpc("agent_start_stream",
                          {"agent_id": a2.id, "message": "go"})
    _wait_for_done(orch, r1["session_id"])
    _wait_for_done(orch, r2["session_id"])

    d1 = orch.handle_rpc("agent_drain_tokens",
                          {"session_id": r1["session_id"]})
    d2 = orch.handle_rpc("agent_drain_tokens",
                          {"session_id": r2["session_id"]})
    assert d1["chunks"] == ["a1-x", "a1-y"]
    assert d2["chunks"] == ["a2-x", "a2-y", "a2-z"]


# ── Helpers ────────────────────────────────────────────────────────────────

def _poll_until(orch, sid: str, *, min_total: int, timeout: float = 2.0) -> None:
    """Wait until at least `min_total` pieces have been deposited into the
    session buffer. Avoids race conditions between `release()` and `drain()`.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Inspect directly (drain would pop — we want to wait, not consume).
        with orch._state_lock:
            have = len(orch._pending_tokens.get(sid, []))
        if have >= min_total:
            return
        time.sleep(0.005)
    raise AssertionError(
        f"Only {have} pieces after {timeout}s (wanted {min_total})"
    )
