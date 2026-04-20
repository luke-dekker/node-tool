"""Graph.snapshot() / revert_to() — transactional graph state.

Used by autoresearch's mutate→eval→keep/revert loop. Hash is content-
addressed and stable across processes for identical graph state; payload
is retained in-memory only (not persisted).
"""
from __future__ import annotations
import pytest

from core.graph import Graph, Connection
from core.node import BaseNode, PortType


class _Widget(BaseNode):
    """Test node with input defaults + one output."""
    type_name = "_test_widget"
    label     = "Widget"

    def _setup_ports(self):
        self.add_input("n", PortType.INT, default=0)
        self.add_input("name", PortType.STRING, default="")
        self.add_output("y", PortType.ANY)

    def execute(self, inputs):
        return {"y": (inputs.get("n"), inputs.get("name"))}


# NODE_REGISTRY needs the test node available for revert_to to reconstruct deleted nodes.
@pytest.fixture(autouse=True)
def _register_widget():
    from nodes import NODE_REGISTRY
    NODE_REGISTRY["_test_widget"] = _Widget
    yield
    NODE_REGISTRY.pop("_test_widget", None)


# ── basic round-trip ──────────────────────────────────────────────────────

def test_snapshot_of_empty_graph_reverts_clean():
    g = Graph()
    h = g.snapshot()
    g.add_node(_Widget())
    assert len(g.nodes) == 1
    g.revert_to(h)
    assert g.nodes == {}
    assert g.connections == []


def test_snapshot_preserves_node_ids_and_defaults():
    g = Graph()
    w = _Widget()
    g.add_node(w)
    w.inputs["n"].default_value = 42
    w.inputs["name"].default_value = "hi"
    h = g.snapshot()

    # Mutate
    w.inputs["n"].default_value = 99
    g.add_node(_Widget())
    assert len(g.nodes) == 2

    g.revert_to(h)
    assert len(g.nodes) == 1
    survivor = g.nodes[w.id]
    assert survivor.inputs["n"].default_value == 42
    assert survivor.inputs["name"].default_value == "hi"


def test_snapshot_preserves_object_identity_for_surviving_nodes():
    g = Graph()
    w = _Widget()
    g.add_node(w)
    h = g.snapshot()
    g.add_node(_Widget())   # add a transient node
    g.revert_to(h)
    # Same Python object still in the graph — only the transient was pruned.
    assert g.nodes[w.id] is w


def test_snapshot_recreates_deleted_nodes():
    g = Graph()
    w1 = _Widget()
    w2 = _Widget()
    g.add_node(w1)
    g.add_node(w2)
    h = g.snapshot()
    g.remove_node(w2.id)
    assert w2.id not in g.nodes
    g.revert_to(h)
    assert w2.id in g.nodes
    # w2 was recreated — it's a NEW object, but with the same id
    assert g.nodes[w2.id] is not w2


def test_snapshot_restores_connections():
    g = Graph()
    a, b = _Widget(), _Widget()
    g.add_node(a)
    g.add_node(b)
    g.add_connection(a.id, "y", b.id, "n")
    h = g.snapshot()

    g.remove_connection(a.id, "y", b.id, "n")
    assert g.connections == []
    g.revert_to(h)
    assert len(g.connections) == 1
    c = g.connections[0]
    assert c.from_node_id == a.id and c.to_node_id == b.id


# ── hash stability ─────────────────────────────────────────────────────────

def test_same_state_gives_same_hash():
    g = Graph()
    w = _Widget()
    g.add_node(w)
    w.inputs["n"].default_value = 7
    h1 = g.snapshot()
    h2 = g.snapshot()
    assert h1 == h2


def test_different_state_gives_different_hash():
    g = Graph()
    w = _Widget()
    g.add_node(w)
    h1 = g.snapshot()
    w.inputs["n"].default_value = 99
    h2 = g.snapshot()
    assert h1 != h2


def test_structural_change_changes_hash():
    g = Graph()
    w = _Widget()
    g.add_node(w)
    h1 = g.snapshot()
    g.add_node(_Widget())
    h2 = g.snapshot()
    assert h1 != h2


def test_connection_order_does_not_affect_hash():
    """Two graphs with connections added in different orders hash identically."""
    g1 = Graph()
    a1, b1 = _Widget(), _Widget()
    g1.nodes[a1.id] = a1
    g1.nodes[b1.id] = b1
    g1.connections.append(Connection(a1.id, "y", b1.id, "n"))
    g1.connections.append(Connection(a1.id, "y", b1.id, "name"))
    h1 = g1.snapshot()

    g2 = Graph()
    g2.nodes[a1.id] = a1   # same ids, different graph object
    g2.nodes[b1.id] = b1
    # Reversed order
    g2.connections.append(Connection(a1.id, "y", b1.id, "name"))
    g2.connections.append(Connection(a1.id, "y", b1.id, "n"))
    h2 = g2.snapshot()
    assert h1 == h2


# ── revert_to edge cases ───────────────────────────────────────────────────

def test_revert_to_unknown_hash_raises():
    g = Graph()
    with pytest.raises(KeyError, match="Unknown snapshot"):
        g.revert_to("deadbeef")


def test_multiple_snapshots_independent():
    g = Graph()
    w = _Widget()
    g.add_node(w)
    w.inputs["n"].default_value = 1
    h1 = g.snapshot()
    w.inputs["n"].default_value = 2
    h2 = g.snapshot()
    w.inputs["n"].default_value = 3

    g.revert_to(h2)
    assert g.nodes[w.id].inputs["n"].default_value == 2
    g.revert_to(h1)
    assert g.nodes[w.id].inputs["n"].default_value == 1


def test_snapshot_skips_non_serializable_defaults():
    """Tensors / objects in input defaults collapse to None in the snapshot,
    so the hash stays stable across runs even when runtime caches change."""
    g = Graph()
    w = _Widget()
    g.add_node(w)
    h1 = g.snapshot()

    class _OpaqueObject:
        pass

    w.inputs["n"].default_value = _OpaqueObject()
    h2 = g.snapshot()
    # The opaque object collapses to None → different from the integer default,
    # so the hash DOES change; but the collapse itself is lossless-in-intent:
    # reverting restores a None default, not the original int 0.
    g.revert_to(h2)
    assert g.nodes[w.id].inputs["n"].default_value is None


# ── revert-after-mutation autoresearch-style flow ──────────────────────────

def test_mutate_eval_revert_cycle():
    """Classic autoresearch flow: snapshot → mutate → eval → revert if bad."""
    g = Graph()
    a, b, c = _Widget(), _Widget(), _Widget()
    for n in (a, b, c):
        g.add_node(n)
    g.add_connection(a.id, "y", b.id, "n")
    g.add_connection(b.id, "y", c.id, "n")
    a.inputs["name"].default_value = "original"
    h_before = g.snapshot()

    # Mutate: rename A's input
    a.inputs["name"].default_value = "mutated"
    # Add a new node
    d = _Widget()
    g.add_node(d)
    g.add_connection(c.id, "y", d.id, "n")
    h_after = g.snapshot()
    assert h_before != h_after

    # "Evaluation failed — revert"
    g.revert_to(h_before)
    assert len(g.nodes) == 3
    assert d.id not in g.nodes
    assert g.nodes[a.id].inputs["name"].default_value == "original"
    # Hash of the reverted state matches the original snapshot hash
    assert g.snapshot() == h_before
