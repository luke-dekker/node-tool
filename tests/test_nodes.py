"""Tests for consolidated and utility node types."""

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from nodes.math import MathNode, ClampNode, MapRangeNode
from nodes.logic import LogicNode, BranchNode
from nodes.string import StringNode, FormatNode, ReplaceNode
from nodes.code import PythonNode
from nodes.data import ConstNode, CastNode, PrintNode
from nodes import NODE_REGISTRY


# ── MathNode ────────────────────────────────────────────────────────────────

def _m(op, a, b=1.0):
    return MathNode().execute({"A": a, "B": b, "Op": op})["Result"]

def test_math_add():        assert _m("add", 1.0, 2.0) == 3.0
def test_math_subtract():   assert _m("subtract", 10.0, 3.0) == 7.0
def test_math_multiply():   assert _m("multiply", 4.0, 5.0) == 20.0
def test_math_divide():     assert _m("divide", 9.0, 3.0) == 3.0
def test_math_divide_zero():assert _m("divide", 5.0, 0.0) == 0.0
def test_math_power():      assert _m("power", 2.0, 10.0) == 1024.0
def test_math_power_zero(): assert _m("power", 100.0, 0.0) == 1.0
def test_math_sqrt():       assert abs(_m("sqrt", 9.0) - 3.0) < 1e-9
def test_math_sqrt_neg():   assert _m("sqrt", -4.0) == 0.0
def test_math_abs_pos():    assert _m("abs", 5.0) == 5.0
def test_math_abs_neg():    assert _m("abs", -5.0) == 5.0
def test_math_sin_zero():   assert abs(_m("sin", 0.0)) < 1e-9
def test_math_sin_90():     assert abs(_m("sin", 90.0) - 1.0) < 1e-9
def test_math_cos_zero():   assert abs(_m("cos", 0.0) - 1.0) < 1e-9
def test_math_cos_90():     assert abs(_m("cos", 90.0)) < 1e-9
def test_math_round():      assert _m("round", 3.14159, 2.0) == 3.14
def test_math_round_zero(): assert _m("round", 3.7, 0.0) == 4.0
def test_math_unknown_op_falls_back():
    assert _m("nonsense", 3.0, 4.0) == 7.0  # falls back to "add"

def test_math_random_range():
    for _ in range(20):
        r = _m("random", 5.0, 10.0)
        assert 5.0 <= r <= 10.0

def test_math_random_reversed():
    for _ in range(10):
        r = _m("random", 10.0, 5.0)
        assert 5.0 <= r <= 10.0


# ── ClampNode ────────────────────────────────────────────────────────────────

def test_clamp_within():
    assert ClampNode().execute({"Value": 0.5, "Min": 0.0, "Max": 1.0})["Result"] == 0.5

def test_clamp_below():
    assert ClampNode().execute({"Value": -1.0, "Min": 0.0, "Max": 1.0})["Result"] == 0.0

def test_clamp_above():
    assert ClampNode().execute({"Value": 5.0, "Min": 0.0, "Max": 1.0})["Result"] == 1.0


# ── MapRangeNode ─────────────────────────────────────────────────────────────

def test_map_range():
    result = MapRangeNode().execute(
        {"Value": 0.5, "InMin": 0.0, "InMax": 1.0, "OutMin": 0.0, "OutMax": 100.0}
    )["Result"]
    assert abs(result - 50.0) < 1e-9

def test_map_range_zero_input_range():
    result = MapRangeNode().execute(
        {"Value": 5.0, "InMin": 5.0, "InMax": 5.0, "OutMin": 10.0, "OutMax": 20.0}
    )["Result"]
    assert result == 10.0


# ── LogicNode ────────────────────────────────────────────────────────────────

def _l(op, a, b=False):
    return LogicNode().execute({"A": a, "B": b, "Op": op})["Result"]

def test_logic_and_true():   assert _l("and", True, True) is True
def test_logic_and_false():  assert _l("and", True, False) is False
def test_logic_or_true():    assert _l("or", False, True) is True
def test_logic_or_false():   assert _l("or", False, False) is False
def test_logic_not_true():   assert _l("not", True) is False
def test_logic_not_false():  assert _l("not", False) is True
def test_logic_eq_true():    assert _l("eq", 3.0, 3.0) is True
def test_logic_eq_false():   assert _l("eq", 1.0, 2.0) is False
def test_logic_lt_true():    assert _l("lt", 1.0, 2.0) is True
def test_logic_gt_true():    assert _l("gt", 5.0, 2.0) is True
def test_logic_neq_true():   assert _l("neq", 1.0, 2.0) is True


# ── BranchNode ───────────────────────────────────────────────────────────────

def test_branch_true():
    assert BranchNode().execute({"Condition": True, "True Value": "yes", "False Value": "no"})["Result"] == "yes"

def test_branch_false():
    assert BranchNode().execute({"Condition": False, "True Value": "yes", "False Value": "no"})["Result"] == "no"


# ── StringNode ───────────────────────────────────────────────────────────────

def _s(op, a, b=""):
    return StringNode().execute({"A": a, "B": b, "Op": op})["Result"]

def test_string_upper():      assert _s("upper", "hello") == "HELLO"
def test_string_lower():      assert _s("lower", "WORLD") == "world"
def test_string_strip():      assert _s("strip", "  hello  ") == "hello"
def test_string_reverse():    assert _s("reverse", "abc") == "cba"
def test_string_length():     assert _s("length", "abcde") == 5
def test_string_length_empty():assert _s("length", "") == 0
def test_string_concat():     assert _s("concat", "hello", " world") == "hello world"
def test_string_contains_true():  assert _s("contains", "hello world", "world") is True
def test_string_contains_false(): assert _s("contains", "hello world", "xyz") is False
def test_string_repeat():     assert _s("repeat", "ab", "3") == "ababab"


# ── FormatNode ───────────────────────────────────────────────────────────────

def test_format():
    result = FormatNode().execute({"Template": "x={0}, y={1}", "Arg0": 10, "Arg1": 20})
    assert result["Result"] == "x=10, y=20"


# ── ReplaceNode ──────────────────────────────────────────────────────────────

def test_replace():
    result = ReplaceNode().execute({"Value": "foo bar foo", "Old": "foo", "New": "baz"})
    assert result["Result"] == "baz bar baz"


# ── PythonNode ───────────────────────────────────────────────────────────────

def test_python_simple_expr():
    result = PythonNode().execute({"a": 3, "b": 4, "c": None, "code": "result = a + b"})
    assert result["result"] == 7

def test_python_string_op():
    result = PythonNode().execute({"a": "hello", "b": None, "c": None, "code": "result = a.upper()"})
    assert result["result"] == "HELLO"

def test_python_uses_math():
    result = PythonNode().execute({"a": 0.0, "b": None, "c": None, "code": "result = math.sin(a)"})
    assert result["result"] == 0.0

def test_python_uses_np():
    result = PythonNode().execute({"a": None, "b": None, "c": None, "code": "result = np.array([1,2,3]).sum()"})
    assert result["result"] == 6

def test_python_bad_code_returns_none():
    result = PythonNode().execute({"a": None, "b": None, "c": None, "code": "result = 1/0"})
    assert result["result"] is None

def test_python_multiline():
    code = "x = a * 2\nresult = x + 1"
    result = PythonNode().execute({"a": 5, "b": None, "c": None, "code": code})
    assert result["result"] == 11


# ── Data nodes ──────────────────────────────────────────────────────────────

def test_print_passthrough():
    result = PrintNode().execute({"Value": 42.0, "Label": "x"})
    assert result["Value"] == 42.0
    assert "42" in result["__terminal__"]
    assert "x" in result["__terminal__"]

def test_print_no_label():
    result = PrintNode().execute({"Value": "hello", "Label": ""})
    assert result["__terminal__"] == "hello"

def test_registry_populated():
    assert len(NODE_REGISTRY) > 0

def test_registry_has_math():
    assert "math" in NODE_REGISTRY

def test_registry_has_python():
    assert "python" in NODE_REGISTRY

def test_registry_has_logic():
    assert "logic" in NODE_REGISTRY

def test_registry_has_print():
    assert "print" in NODE_REGISTRY

def test_registry_instantiable():
    for type_name, cls in NODE_REGISTRY.items():
        node = cls()
        assert node.type_name == type_name
        assert isinstance(node.inputs, dict)
        assert isinstance(node.outputs, dict)
