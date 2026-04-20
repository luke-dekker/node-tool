"""Graph.subgraph_between — topological cone between two node ids.

Used by GraphAsToolNode (now) and autoresearch's mutator-scope gate (Phase C).
Algorithm: descendants(A) ∩ ancestors(B), both inclusive, sorted
topologically.
"""
from __future__ import annotations
import pytest

from core.graph import Graph, Connection
from core.node import BaseNode, PortType


class _Passthrough(BaseNode):
    type_name = "_test_passthrough"
    label     = "Pass"

    def _setup_ports(self):
        self.add_input("x", PortType.ANY)
        self.add_output("x", PortType.ANY)

    def execute(self, inputs):
        return {"x": inputs.get("x")}


def _line_graph(n: int) -> tuple[Graph, list[str]]:
    """Return a Graph of `n` passthrough nodes connected in a line."""
    g = Graph()
    nodes = [_Passthrough() for _ in range(n)]
    for node in nodes:
        g.add_node(node)
    for i in range(n - 1):
        g.add_connection(nodes[i].id, "x", nodes[i + 1].id, "x")
    return g, [n.id for n in nodes]


# ── basic cones ────────────────────────────────────────────────────────────

def test_subgraph_single_node_returns_itself():
    g, ids = _line_graph(1)
    assert g.subgraph_between(ids[0], ids[0]) == [ids[0]]


def test_subgraph_self_inclusive_in_line():
    g, ids = _line_graph(4)
    sub = g.subgraph_between(ids[1], ids[1])
    assert sub == [ids[1]]


def test_subgraph_line_end_to_end():
    g, ids = _line_graph(4)
    sub = g.subgraph_between(ids[0], ids[3])
    assert sub == ids  # full line


def test_subgraph_line_interior():
    g, ids = _line_graph(5)
    sub = g.subgraph_between(ids[1], ids[3])
    assert sub == ids[1:4]


def test_subgraph_wrong_direction_empty():
    """B ahead of A in topo order — OK. A ahead of B reversed — empty."""
    g, ids = _line_graph(3)
    assert g.subgraph_between(ids[2], ids[0]) == []


# ── disconnected subgraph ──────────────────────────────────────────────────

def test_subgraph_disconnected_components():
    g, ids = _line_graph(2)
    extra = _Passthrough()
    g.add_node(extra)
    assert g.subgraph_between(ids[0], extra.id) == []
    assert g.subgraph_between(extra.id, ids[1]) == []


# ── diamond (cone is full diamond) ─────────────────────────────────────────

def test_subgraph_diamond_includes_both_branches():
    """a → b, a → c, b → d, c → d. Cone a↔d is {a, b, c, d}."""
    g = Graph()
    a, b, c, d = (_Passthrough() for _ in range(4))
    for n in (a, b, c, d):
        g.add_node(n)
    # a has two outputs via multiple connections (many-to-one on inputs);
    # we need multiple outputs per node. Use a helper node with two outs.

    # Simpler: use multiple downstream nodes; branches reconverge on 'd'
    # with the latest connection winning on 'x' (d has just one 'x' input).
    # To properly test a diamond we need d to accept two inputs. Use a
    # Two-input passthrough.

    class _TwoIn(BaseNode):
        type_name = "_test_two_in"
        label     = "TwoIn"
        def _setup_ports(self):
            self.add_input("a", PortType.ANY)
            self.add_input("b", PortType.ANY)
            self.add_output("x", PortType.ANY)
        def execute(self, inputs):
            return {"x": (inputs.get("a"), inputs.get("b"))}

    d2 = _TwoIn()
    g.nodes.pop(d.id)   # remove plain d
    g.add_node(d2)
    g.add_connection(a.id, "x", b.id, "x")
    g.add_connection(a.id, "x", c.id, "x")
    g.add_connection(b.id, "x", d2.id, "a")
    g.add_connection(c.id, "x", d2.id, "b")

    cone = g.subgraph_between(a.id, d2.id)
    assert set(cone) == {a.id, b.id, c.id, d2.id}
    # Topological ordering: a first, d2 last, b and c in between.
    assert cone[0] == a.id
    assert cone[-1] == d2.id


# ── node exclusion (outside cone) ──────────────────────────────────────────

def test_subgraph_excludes_sibling_not_on_path():
    """a → b; a → c (c is NOT ancestor of b). b = target; c must be excluded."""
    g = Graph()
    a, b, c = (_Passthrough() for _ in range(3))
    for n in (a, b, c):
        g.add_node(n)
    g.add_connection(a.id, "x", b.id, "x")
    g.add_connection(a.id, "x", c.id, "x")
    cone = g.subgraph_between(a.id, b.id)
    assert set(cone) == {a.id, b.id}
    assert c.id not in cone


# ── error cases ────────────────────────────────────────────────────────────

def test_subgraph_unknown_node_raises():
    g, ids = _line_graph(2)
    with pytest.raises(KeyError, match="not found"):
        g.subgraph_between("nope", ids[0])
    with pytest.raises(KeyError, match="not found"):
        g.subgraph_between(ids[0], "nope")
