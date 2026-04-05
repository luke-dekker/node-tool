"""Tests for core graph and execution logic."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.graph import Graph
from nodes.math import AddNode, MultiplyNode, DivideNode
from nodes.data import FloatConstNode, PrintNode, IntConstNode
from nodes.logic import CompareNode, BranchNode


def test_add_node():
    node = AddNode()
    result = node.execute({"A": 3.0, "B": 4.0})
    assert result["Result"] == 7.0


def test_add_node_negative():
    node = AddNode()
    result = node.execute({"A": -10.0, "B": 3.0})
    assert result["Result"] == -7.0


def test_multiply_node():
    node = MultiplyNode()
    result = node.execute({"A": 6.0, "B": 7.0})
    assert result["Result"] == 42.0


def test_divide_by_zero():
    node = DivideNode()
    result = node.execute({"A": 10.0, "B": 0.0})
    assert result["Result"] == 0.0  # should not raise


def test_divide_normal():
    node = DivideNode()
    result = node.execute({"A": 10.0, "B": 4.0})
    assert result["Result"] == 2.5


def test_graph_single_node():
    g = Graph()
    n = g.add_node(FloatConstNode())
    n.inputs["Value"].default_value = 42.0
    outputs, terminal = g.execute()
    assert outputs[n.id]["Value"] == 42.0


def test_graph_execution():
    """Wire two FloatConst nodes into an Add, Add into Print, verify terminal output."""
    g = Graph()

    c1 = g.add_node(FloatConstNode())
    c1.inputs["Value"].default_value = 10.0

    c2 = g.add_node(FloatConstNode())
    c2.inputs["Value"].default_value = 5.0

    add = g.add_node(AddNode())

    printer = g.add_node(PrintNode())
    printer.inputs["Label"].default_value = "Result"

    g.add_connection(c1.id, "Value", add.id, "A")
    g.add_connection(c2.id, "Value", add.id, "B")
    g.add_connection(add.id, "Result", printer.id, "Value")

    outputs, terminal = g.execute()
    assert outputs[add.id]["Result"] == 15.0
    assert len(terminal) == 1
    assert "15" in terminal[0]


def test_topological_sort():
    """Ensure graph executes in correct order (no node uses value before it's computed)."""
    g = Graph()
    order_log: list[str] = []

    c1 = g.add_node(FloatConstNode())
    c1.inputs["Value"].default_value = 3.0

    c2 = g.add_node(FloatConstNode())
    c2.inputs["Value"].default_value = 4.0

    add = g.add_node(AddNode())

    mul = g.add_node(MultiplyNode())

    g.add_connection(c1.id, "Value", add.id, "A")
    g.add_connection(c2.id, "Value", add.id, "B")
    g.add_connection(add.id, "Result", mul.id, "A")

    mul.inputs["B"].default_value = 2.0

    outputs, _ = g.execute()
    assert outputs[add.id]["Result"] == 7.0
    assert outputs[mul.id]["Result"] == 14.0


def test_cycle_detection():
    """Adding a connection that creates a cycle should return None."""
    g = Graph()
    a = g.add_node(AddNode())
    b = g.add_node(AddNode())

    conn1 = g.add_connection(a.id, "Result", b.id, "A")
    assert conn1 is not None

    # This would create a cycle: b -> a when a -> b already exists
    conn2 = g.add_connection(b.id, "Result", a.id, "A")
    assert conn2 is None  # cycle detected, not added


def test_graph_execute():
    g = Graph()
    c = g.add_node(FloatConstNode())
    c.inputs["Value"].default_value = 99.0

    outputs, terminal = g.execute()
    assert outputs[c.id]["Value"] == 99.0


def test_remove_node_cleans_connections():
    g = Graph()
    a = g.add_node(FloatConstNode())
    b = g.add_node(AddNode())
    g.add_connection(a.id, "Value", b.id, "A")
    assert len(g.connections) == 1

    g.remove_node(a.id)
    assert len(g.connections) == 0
    assert a.id not in g.nodes


def test_missing_connection_uses_default():
    """A disconnected input should use its default value."""
    g = Graph()
    add = g.add_node(AddNode())
    add.inputs["A"].default_value = 7.0
    add.inputs["B"].default_value = 3.0

    outputs, _ = g.execute()
    assert outputs[add.id]["Result"] == 10.0


def test_compare_node():
    node = CompareNode()
    # op=4 means greater
    result = node.execute({"A": 5.0, "B": 3.0, "Op": 4})
    assert result["Result"] is True
    result2 = node.execute({"A": 2.0, "B": 3.0, "Op": 4})
    assert result2["Result"] is False


def test_branch_node():
    node = BranchNode()
    result = node.execute({"Condition": True, "True Value": 100.0, "False Value": 0.0})
    assert result["Result"] == 100.0
    result2 = node.execute({"Condition": False, "True Value": 100.0, "False Value": 0.0})
    assert result2["Result"] == 0.0


def test_long_chain():
    """FloatConst -> Add -> Multiply -> Add chain."""
    g = Graph()
    c = g.add_node(FloatConstNode())
    c.inputs["Value"].default_value = 5.0

    add1 = g.add_node(AddNode())
    add1.inputs["B"].default_value = 5.0  # 5+5=10

    mul = g.add_node(MultiplyNode())
    mul.inputs["B"].default_value = 3.0  # 10*3=30

    add2 = g.add_node(AddNode())
    add2.inputs["B"].default_value = 2.0  # 30+2=32

    g.add_connection(c.id, "Value", add1.id, "A")
    g.add_connection(add1.id, "Result", mul.id, "A")
    g.add_connection(mul.id, "Result", add2.id, "A")

    outputs, _ = g.execute()
    assert outputs[add2.id]["Result"] == 32.0
