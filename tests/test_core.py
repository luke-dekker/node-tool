"""Tests for core graph and execution logic."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.graph import Graph
from nodes.math import MathNode, ClampNode
from nodes.data import ConstNode, PrintNode
from nodes.logic import LogicNode, BranchNode


# ── Node execute() sanity checks ─────────────────────────────────────────────

def test_add_node():
    assert MathNode().execute({"A": 3.0, "B": 4.0, "Op": "add"})["Result"] == 7.0

def test_add_node_negative():
    assert MathNode().execute({"A": -10.0, "B": 3.0, "Op": "add"})["Result"] == -7.0

def test_multiply_node():
    assert MathNode().execute({"A": 6.0, "B": 7.0, "Op": "multiply"})["Result"] == 42.0

def test_divide_by_zero():
    assert MathNode().execute({"A": 10.0, "B": 0.0, "Op": "divide"})["Result"] == 0.0

def test_divide_normal():
    assert MathNode().execute({"A": 10.0, "B": 4.0, "Op": "divide"})["Result"] == 2.5


# ── Graph execution ──────────────────────────────────────────────────────────

def test_graph_single_node():
    g = Graph()
    n = g.add_node(ConstNode())
    n.inputs["Value"].default_value = 42.0
    outputs, terminal = g.execute()
    assert outputs[n.id]["Value"] == 42.0


def test_graph_execution():
    """Wire two FloatConst nodes into MathNode(add), into Print, verify output."""
    g = Graph()
    c1 = g.add_node(ConstNode())
    c1.inputs["Value"].default_value = 10.0
    c2 = g.add_node(ConstNode())
    c2.inputs["Value"].default_value = 5.0

    add = g.add_node(MathNode())  # default Op="add"
    printer = g.add_node(PrintNode())
    printer.inputs["Label"].default_value = "Result"

    g.add_connection(c1.id, "Value", add.id, "A")
    g.add_connection(c2.id, "Value", add.id, "B")
    g.add_connection(add.id, "Result", printer.id, "Value")

    outputs, terminal = g.execute()
    assert outputs[add.id]["Result"] == 15.0
    assert "15" in terminal[0]


def test_topological_sort():
    """Verify graph executes in dependency order."""
    g = Graph()
    c1 = g.add_node(ConstNode()); c1.inputs["Value"].default_value = 3.0
    c2 = g.add_node(ConstNode()); c2.inputs["Value"].default_value = 4.0

    add = g.add_node(MathNode())   # add A+B
    mul = g.add_node(MathNode())   # multiply A*B
    mul.inputs["Op"].default_value = "multiply"
    mul.inputs["B"].default_value = 2.0

    g.add_connection(c1.id, "Value", add.id, "A")
    g.add_connection(c2.id, "Value", add.id, "B")
    g.add_connection(add.id, "Result", mul.id, "A")

    outputs, _ = g.execute()
    assert outputs[add.id]["Result"] == 7.0
    assert outputs[mul.id]["Result"] == 14.0


def test_cycle_detection():
    """Adding a back-edge should return None (cycle rejected)."""
    g = Graph()
    a = g.add_node(MathNode())
    b = g.add_node(MathNode())
    assert g.add_connection(a.id, "Result", b.id, "A") is not None
    assert g.add_connection(b.id, "Result", a.id, "A") is None  # cycle


def test_graph_execute():
    g = Graph()
    c = g.add_node(ConstNode())
    c.inputs["Value"].default_value = 99.0
    outputs, _ = g.execute()
    assert outputs[c.id]["Value"] == 99.0


def test_remove_node_cleans_connections():
    g = Graph()
    a = g.add_node(ConstNode())
    b = g.add_node(MathNode())
    g.add_connection(a.id, "Value", b.id, "A")
    assert len(g.connections) == 1
    g.remove_node(a.id)
    assert len(g.connections) == 0
    assert a.id not in g.nodes


def test_missing_connection_uses_default():
    g = Graph()
    add = g.add_node(MathNode())
    add.inputs["A"].default_value = 7.0
    add.inputs["B"].default_value = 3.0
    outputs, _ = g.execute()
    assert outputs[add.id]["Result"] == 10.0


def test_logic_node():
    assert LogicNode().execute({"A": 5.0, "B": 3.0, "Op": "gt"})["Result"] is True
    assert LogicNode().execute({"A": 2.0, "B": 3.0, "Op": "gt"})["Result"] is False


def test_branch_node():
    node = BranchNode()
    assert node.execute({"Condition": True,  "True Value": 100.0, "False Value": 0.0})["Result"] == 100.0
    assert node.execute({"Condition": False, "True Value": 100.0, "False Value": 0.0})["Result"] == 0.0


def test_long_chain():
    """FloatConst -> add -> multiply -> add chain: ((5+5)*3)+2 = 32."""
    g = Graph()
    c = g.add_node(ConstNode()); c.inputs["Value"].default_value = 5.0

    add1 = g.add_node(MathNode()); add1.inputs["B"].default_value = 5.0
    mul  = g.add_node(MathNode()); mul.inputs["Op"].default_value = "multiply"; mul.inputs["B"].default_value = 3.0
    add2 = g.add_node(MathNode()); add2.inputs["B"].default_value = 2.0

    g.add_connection(c.id,    "Value",  add1.id, "A")
    g.add_connection(add1.id, "Result", mul.id,  "A")
    g.add_connection(mul.id,  "Result", add2.id, "A")

    outputs, _ = g.execute()
    assert outputs[add2.id]["Result"] == 32.0
