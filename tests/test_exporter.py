"""Tests for the graph exporter — headless, no DPG."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.graph import Graph
from core.exporter import GraphExporter


def _export(graph):
    return GraphExporter().export(graph)


# ── Basic plumbing ────────────────────────────────────────────────────────────

def test_empty_graph():
    script = _export(Graph())
    assert "Empty graph" in script


def test_single_float_const():
    from nodes.data import ConstNode
    g = Graph()
    n = ConstNode()
    n.inputs["Value"].default_value = 3.14
    g.add_node(n)
    script = _export(g)
    assert "3.14" in script
    assert "const" in script


def test_add_chain():
    from nodes.data import ConstNode
    from nodes.math import MathNode
    g = Graph()
    a = ConstNode(); a.inputs["Value"].default_value = 1.0
    b = ConstNode(); b.inputs["Value"].default_value = 2.0
    add = MathNode()
    g.add_node(a); g.add_node(b); g.add_node(add)
    g.add_connection(a.id, "Value", add.id, "A")
    g.add_connection(b.id, "Value", add.id, "B")
    script = _export(g)
    assert "const" in script
    assert "math" in script


def test_print_node():
    from nodes.data import ConstNode, PrintNode
    g = Graph()
    c = ConstNode(); c.inputs["Value"].default_value = 42.0
    p = PrintNode()
    g.add_node(c); g.add_node(p)
    g.add_connection(c.id, "Value", p.id, "Value")
    script = _export(g)
    assert "print(" in script


# ── Math nodes ────────────────────────────────────────────────────────────────

def test_math_nodes():
    from nodes.math import MathNode, ClampNode, MapRangeNode
    for NodeCls in [MathNode, ClampNode, MapRangeNode]:
        g = Graph()
        g.add_node(NodeCls())
        script = _export(g)
        assert "export not" not in script, f"{NodeCls.__name__} unsupported"


# ── Logic nodes ───────────────────────────────────────────────────────────────

def test_logic_nodes():
    from nodes.logic import LogicNode, BranchNode
    for NodeCls in [LogicNode, BranchNode]:
        g = Graph()
        g.add_node(NodeCls())
        script = _export(g)
        assert "export not" not in script, f"{NodeCls.__name__} unsupported"


# ── String nodes ──────────────────────────────────────────────────────────────

def test_string_nodes():
    from nodes.string import StringNode, ReplaceNode, FormatNode
    for NodeCls in [StringNode, ReplaceNode, FormatNode]:
        g = Graph()
        g.add_node(NodeCls())
        script = _export(g)
        assert "export not" not in script, f"{NodeCls.__name__} unsupported"


# ── NumPy nodes ───────────────────────────────────────────────────────────────

def test_numpy_nodes():
    import nodes.numpy as nm
    for NodeCls in [nm.NpArangeNode, nm.NpLinspaceNode, nm.NpZerosNode, nm.NpOnesNode,
                    nm.NpRandNode, nm.NpRandnNode, nm.NpEyeNode, nm.NpMeanNode,
                    nm.NpStdNode, nm.NpSumNode, nm.NpMinNode, nm.NpMaxNode,
                    nm.NpAbsNode, nm.NpSqrtNode, nm.NpLogNode, nm.NpExpNode,
                    nm.NpClipNode, nm.NpReshapeNode, nm.NpTransposeNode,
                    nm.NpFlattenNode, nm.NpDotNode, nm.NpMatMulNode, nm.NpInvNode]:
        g = Graph()
        g.add_node(NodeCls())
        script = _export(g)
        assert "export not" not in script, f"{NodeCls.__name__} unsupported"
    assert "import numpy as np" in _export(_build_single(nm.NpArangeNode))


def _build_single(NodeCls):
    g = Graph()
    g.add_node(NodeCls())
    return g


# ── Pandas nodes ──────────────────────────────────────────────────────────────

def test_pandas_nodes():
    import nodes.pandas as pn
    for NodeCls in [pn.PdFromCsvNode, pn.PdDescribeNode, pn.PdDropNaNode,
                    pn.PdFillNaNode, pn.PdSortNode, pn.PdResetIndexNode,
                    pn.PdToNumpyNode, pn.PdGroupByNode, pn.PdCorrelationNode,
                    pn.PdXYSplitNode, pn.PdSelectColsNode, pn.PdDropColsNode,
                    pn.PdGetColumnNode]:
        g = Graph()
        g.add_node(NodeCls())
        script = _export(g)
        assert "export not" not in script, f"{NodeCls.__name__} unsupported"


# ── Sklearn nodes ─────────────────────────────────────────────────────────────

def test_sklearn_nodes():
    import nodes.sklearn as sk
    for NodeCls in [sk.SkTrainTestSplitNode, sk.SkStandardScalerNode,
                    sk.SkMinMaxScalerNode, sk.SkLinearRegressionNode,
                    sk.SkLogisticRegressionNode, sk.SkRandomForestNode,
                    sk.SkSVCNode, sk.SkPredictNode, sk.SkAccuracyNode,
                    sk.SkR2ScoreNode, sk.SkCrossValScoreNode,
                    sk.SkKMeansNode, sk.SkPCANode]:
        g = Graph()
        g.add_node(NodeCls())
        script = _export(g)
        assert "export not" not in script, f"{NodeCls.__name__} unsupported"


# ── PyTorch nodes ─────────────────────────────────────────────────────────────

def test_pytorch_layer_nodes():
    import nodes.pytorch as pt
    for NodeCls in [pt.LinearNode, pt.Conv2dNode, pt.BatchNorm1dNode,
                    pt.ActivationNode, pt.DropoutNode, pt.FlattenNode,
                    pt.MaxPool2dNode, pt.AvgPool2dNode,
                    pt.MSELossNode, pt.CrossEntropyLossNode, pt.BCELossNode,
                    pt.BCEWithLogitsNode, pt.L1LossNode]:
        g = Graph()
        g.add_node(NodeCls())
        script = _export(g)
        assert "import torch" in script, f"{NodeCls.__name__} missing torch import"


def test_scheduler_nodes_export():
    import nodes.pytorch as pt
    for NodeCls in [pt.StepLRNode, pt.MultiStepLRNode, pt.ExponentialLRNode,
                    pt.CosineAnnealingLRNode, pt.ReduceLROnPlateauNode]:
        g = Graph()
        g.add_node(NodeCls())
        script = _export(g)
        assert "lr_scheduler" in script, f"{NodeCls.__name__} missing scheduler import"


def test_imports_deduplicated():
    from nodes.pytorch import LinearNode, ActivationNode
    g = Graph()
    g.add_node(LinearNode()); g.add_node(ActivationNode())
    script = _export(g)
    assert script.count("import torch.nn as nn") == 1


def test_script_is_valid_python():
    """The generated script should at least be parseable by ast."""
    import ast
    from nodes.data import ConstNode, PrintNode
    from nodes.math import MathNode
    g = Graph()
    a = ConstNode(); b = ConstNode(); add = MathNode(); pr = PrintNode()
    g.add_node(a); g.add_node(b); g.add_node(add); g.add_node(pr)
    g.add_connection(a.id, "Value", add.id, "A")
    g.add_connection(b.id, "Value", add.id, "B")
    g.add_connection(add.id, "Result", pr.id, "Value")
    script = _export(g)
    ast.parse(script)  # should not raise
