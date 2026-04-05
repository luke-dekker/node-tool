"""Tests for every node type."""

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from nodes.math import (
    AddNode, SubtractNode, MultiplyNode, DivideNode,
    PowerNode, ClampNode, MapRangeNode, RoundNode,
    AbsNode, SqrtNode, SinNode, CosNode, RandomFloatNode,
)
from nodes.logic import CompareNode, AndNode, OrNode, NotNode, BranchNode
from nodes.string import (
    ConcatNode, FormatNode, UpperNode, LowerNode,
    StripNode, LengthNode, ContainsNode, ReplaceNode,
)
from nodes.data import (
    FloatConstNode, IntConstNode, BoolConstNode, StringConstNode,
    PrintNode, ToFloatNode, ToIntNode, ToStringNode, ToBoolNode,
)
from nodes import NODE_REGISTRY


# ── Math nodes ─────────────────────────────────────────────────────────────

def test_add():
    assert AddNode().execute({"A": 1.0, "B": 2.0})["Result"] == 3.0

def test_subtract():
    assert SubtractNode().execute({"A": 10.0, "B": 3.0})["Result"] == 7.0

def test_multiply():
    assert MultiplyNode().execute({"A": 4.0, "B": 5.0})["Result"] == 20.0

def test_divide():
    assert DivideNode().execute({"A": 9.0, "B": 3.0})["Result"] == 3.0

def test_divide_by_zero_safe():
    assert DivideNode().execute({"A": 5.0, "B": 0.0})["Result"] == 0.0

def test_power():
    assert PowerNode().execute({"Base": 2.0, "Exp": 10.0})["Result"] == 1024.0

def test_power_zero_exp():
    assert PowerNode().execute({"Base": 100.0, "Exp": 0.0})["Result"] == 1.0

def test_clamp_within():
    assert ClampNode().execute({"Value": 0.5, "Min": 0.0, "Max": 1.0})["Result"] == 0.5

def test_clamp_below():
    assert ClampNode().execute({"Value": -1.0, "Min": 0.0, "Max": 1.0})["Result"] == 0.0

def test_clamp_above():
    assert ClampNode().execute({"Value": 5.0, "Min": 0.0, "Max": 1.0})["Result"] == 1.0

def test_map_range():
    result = MapRangeNode().execute({"Value": 0.5, "InMin": 0.0, "InMax": 1.0,
                                      "OutMin": 0.0, "OutMax": 100.0})["Result"]
    assert abs(result - 50.0) < 1e-9

def test_map_range_zero_input_range():
    result = MapRangeNode().execute({"Value": 5.0, "InMin": 5.0, "InMax": 5.0,
                                      "OutMin": 10.0, "OutMax": 20.0})["Result"]
    assert result == 10.0  # fallback to out_min

def test_round():
    assert RoundNode().execute({"Value": 3.14159, "Decimals": 2})["Result"] == 3.14

def test_round_zero_decimals():
    assert RoundNode().execute({"Value": 3.7, "Decimals": 0})["Result"] == 4.0

def test_abs_positive():
    assert AbsNode().execute({"Value": 5.0})["Result"] == 5.0

def test_abs_negative():
    assert AbsNode().execute({"Value": -5.0})["Result"] == 5.0

def test_sqrt():
    assert abs(SqrtNode().execute({"Value": 9.0})["Result"] - 3.0) < 1e-9

def test_sqrt_negative_safe():
    assert SqrtNode().execute({"Value": -4.0})["Result"] == 0.0

def test_sin_zero():
    assert abs(SinNode().execute({"Degrees": 0.0})["Result"]) < 1e-9

def test_sin_90():
    assert abs(SinNode().execute({"Degrees": 90.0})["Result"] - 1.0) < 1e-9

def test_cos_zero():
    assert abs(CosNode().execute({"Degrees": 0.0})["Result"] - 1.0) < 1e-9

def test_cos_90():
    assert abs(CosNode().execute({"Degrees": 90.0})["Result"]) < 1e-9

def test_random_float_range():
    for _ in range(20):
        r = RandomFloatNode().execute({"Min": 5.0, "Max": 10.0})["Result"]
        assert 5.0 <= r <= 10.0

def test_random_float_reversed_range():
    for _ in range(10):
        r = RandomFloatNode().execute({"Min": 10.0, "Max": 5.0})["Result"]
        assert 5.0 <= r <= 10.0


# ── Logic nodes ────────────────────────────────────────────────────────────

def test_compare_equal():
    assert CompareNode().execute({"A": 3.0, "B": 3.0, "Op": 2})["Result"] is True

def test_compare_less():
    assert CompareNode().execute({"A": 1.0, "B": 2.0, "Op": 0})["Result"] is True

def test_compare_greater():
    assert CompareNode().execute({"A": 5.0, "B": 2.0, "Op": 4})["Result"] is True

def test_compare_not_equal():
    assert CompareNode().execute({"A": 1.0, "B": 2.0, "Op": 5})["Result"] is True

def test_and_true():
    assert AndNode().execute({"A": True, "B": True})["Result"] is True

def test_and_false():
    assert AndNode().execute({"A": True, "B": False})["Result"] is False

def test_or_true():
    assert OrNode().execute({"A": False, "B": True})["Result"] is True

def test_or_false():
    assert OrNode().execute({"A": False, "B": False})["Result"] is False

def test_not_true():
    assert NotNode().execute({"Value": True})["Result"] is False

def test_not_false():
    assert NotNode().execute({"Value": False})["Result"] is True

def test_branch_true():
    result = BranchNode().execute({"Condition": True, "True Value": "yes", "False Value": "no"})
    assert result["Result"] == "yes"

def test_branch_false():
    result = BranchNode().execute({"Condition": False, "True Value": "yes", "False Value": "no"})
    assert result["Result"] == "no"


# ── String nodes ───────────────────────────────────────────────────────────

def test_concat():
    assert ConcatNode().execute({"A": "hello", "B": " world"})["Result"] == "hello world"

def test_format():
    result = FormatNode().execute({"Template": "x={0}, y={1}", "Arg0": 10, "Arg1": 20})
    assert result["Result"] == "x=10, y=20"

def test_upper():
    assert UpperNode().execute({"Value": "hello"})["Result"] == "HELLO"

def test_lower():
    assert LowerNode().execute({"Value": "WORLD"})["Result"] == "world"

def test_strip():
    assert StripNode().execute({"Value": "  hello  "})["Result"] == "hello"

def test_length():
    assert LengthNode().execute({"Value": "abcde"})["Length"] == 5

def test_length_empty():
    assert LengthNode().execute({"Value": ""})["Length"] == 0

def test_contains_true():
    assert ContainsNode().execute({"Haystack": "hello world", "Needle": "world"})["Result"] is True

def test_contains_false():
    assert ContainsNode().execute({"Haystack": "hello world", "Needle": "xyz"})["Result"] is False

def test_replace():
    result = ReplaceNode().execute({"Value": "foo bar foo", "Old": "foo", "New": "baz"})
    assert result["Result"] == "baz bar baz"


# ── Data nodes ─────────────────────────────────────────────────────────────

def test_float_const():
    assert FloatConstNode().execute({"Value": 3.14})["Value"] == 3.14

def test_int_const():
    assert IntConstNode().execute({"Value": 42})["Value"] == 42

def test_bool_const_true():
    assert BoolConstNode().execute({"Value": True})["Value"] is True

def test_bool_const_false():
    assert BoolConstNode().execute({"Value": False})["Value"] is False

def test_string_const():
    assert StringConstNode().execute({"Value": "node-tool"})["Value"] == "node-tool"

def test_print_passthrough():
    result = PrintNode().execute({"Value": 42.0, "Label": "x"})
    assert result["Value"] == 42.0
    assert "42" in result["__terminal__"]
    assert "x" in result["__terminal__"]

def test_print_no_label():
    result = PrintNode().execute({"Value": "hello", "Label": ""})
    assert result["__terminal__"] == "hello"

def test_to_float():
    assert ToFloatNode().execute({"Value": "3.14"})["Result"] == 3.14

def test_to_float_invalid():
    assert ToFloatNode().execute({"Value": "abc"})["Result"] == 0.0

def test_to_int():
    assert ToIntNode().execute({"Value": 3.9})["Result"] == 3

def test_to_int_string():
    assert ToIntNode().execute({"Value": "7"})["Result"] == 7

def test_to_string():
    assert ToStringNode().execute({"Value": 123})["Result"] == "123"

def test_to_bool_zero():
    assert ToBoolNode().execute({"Value": 0})["Result"] is False

def test_to_bool_nonzero():
    assert ToBoolNode().execute({"Value": 1})["Result"] is True

def test_to_bool_string_false():
    assert ToBoolNode().execute({"Value": "false"})["Result"] is False

def test_to_bool_string_true():
    assert ToBoolNode().execute({"Value": "yes"})["Result"] is True


# ── Registry ───────────────────────────────────────────────────────────────

def test_registry_populated():
    assert len(NODE_REGISTRY) > 0

def test_registry_has_add():
    assert "add" in NODE_REGISTRY

def test_registry_has_print():
    assert "print" in NODE_REGISTRY

def test_registry_instantiable():
    for type_name, cls in NODE_REGISTRY.items():
        node = cls()
        assert node.type_name == type_name
        assert isinstance(node.inputs, dict)
        assert isinstance(node.outputs, dict)
