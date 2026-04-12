"""Tests for core graph and execution logic using PythonNode."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.graph import Graph
from nodes.code.python_node import PythonNode


def _py(code="result = a"):
    n = PythonNode()
    n.inputs["code"].default_value = code
    return n


def _const(value):
    n = PythonNode()
    n.inputs["code"].default_value = f"result = {value!r}"
    return n


# ── PythonNode execute() ────────────────────────────────────────────────────

def test_python_add():
    n = _py("result = a + b")
    assert n.execute({"a": 3.0, "b": 4.0, "c": None, "code": "result = a + b"})["result"] == 7.0

def test_python_multiply():
    n = _py("result = a * b")
    assert n.execute({"a": 6.0, "b": 7.0, "c": None, "code": "result = a * b"})["result"] == 42.0


# ── Graph execution ──────────────────────────────────────────────────────────

def test_graph_single_node():
    g = Graph()
    n = g.add_node(_const(42.0))
    outputs, _ = g.execute()
    assert outputs[n.id]["result"] == 42.0


def test_graph_execution():
    g = Graph()
    c1 = g.add_node(_const(10.0))
    c2 = g.add_node(_const(5.0))
    add = g.add_node(_py("result = a + b"))
    g.add_connection(c1.id, "result", add.id, "a")
    g.add_connection(c2.id, "result", add.id, "b")
    outputs, _ = g.execute()
    assert outputs[add.id]["result"] == 15.0


def test_topological_sort():
    g = Graph()
    c1 = g.add_node(_const(3.0))
    c2 = g.add_node(_const(4.0))
    add = g.add_node(_py("result = a + b"))
    mul = g.add_node(_py("result = a * 2"))
    g.add_connection(c1.id, "result", add.id, "a")
    g.add_connection(c2.id, "result", add.id, "b")
    g.add_connection(add.id, "result", mul.id, "a")
    outputs, _ = g.execute()
    assert outputs[add.id]["result"] == 7.0
    assert outputs[mul.id]["result"] == 14.0


def test_cycle_detection():
    g = Graph()
    a = g.add_node(_py("result = a"))
    b = g.add_node(_py("result = a"))
    assert g.add_connection(a.id, "result", b.id, "a") is not None
    assert g.add_connection(b.id, "result", a.id, "a") is None  # cycle


def test_remove_node_cleans_connections():
    g = Graph()
    a = g.add_node(_const(1.0))
    b = g.add_node(_py("result = a + 1"))
    g.add_connection(a.id, "result", b.id, "a")
    assert len(g.connections) == 1
    g.remove_node(a.id)
    assert len(g.connections) == 0
    assert a.id not in g.nodes


def test_missing_connection_uses_default():
    g = Graph()
    n = g.add_node(_py("result = (a or 7) + (b or 3)"))
    outputs, _ = g.execute()
    assert outputs[n.id]["result"] == 10.0


def test_long_chain():
    """Const(5) -> +5 -> *3 -> +2 = 32."""
    g = Graph()
    c = g.add_node(_const(5.0))
    add1 = g.add_node(_py("result = a + 5"))
    mul  = g.add_node(_py("result = a * 3"))
    add2 = g.add_node(_py("result = a + 2"))
    g.add_connection(c.id,    "result", add1.id, "a")
    g.add_connection(add1.id, "result", mul.id,  "a")
    g.add_connection(mul.id,  "result", add2.id, "a")
    outputs, _ = g.execute()
    assert outputs[add2.id]["result"] == 32.0
