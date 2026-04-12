"""Tests for the graph exporter — headless, no DPG."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ast
import pytest
from core.graph import Graph
from core.exporter import GraphExporter
from nodes.code.python_node import PythonNode


def _export(graph):
    return GraphExporter().export(graph)


def _py(code="result = a"):
    n = PythonNode()
    n.inputs["code"].default_value = code
    return n


def _const(value):
    return _py(f"result = {value!r}")


# ── Basic plumbing ────────────────────────────────────────────────────────────

def test_empty_graph():
    script = _export(Graph())
    assert "Empty graph" in script


def test_single_const():
    g = Graph()
    g.add_node(_const(3.14))
    script = _export(g)
    assert "3.14" in script


def test_chain():
    g = Graph()
    a = _const(1.0); b = _const(2.0); add = _py("result = a + b")
    g.add_node(a); g.add_node(b); g.add_node(add)
    g.add_connection(a.id, "result", add.id, "a")
    g.add_connection(b.id, "result", add.id, "b")
    script = _export(g)
    assert "result" in script


# ── NumPy nodes ───────────────────────────────────────────────────────────────

def _build_single(NodeCls):
    g = Graph()
    g.add_node(NodeCls())
    return g


def test_numpy_nodes():
    import nodes.numpy as nm
    for NodeCls in [nm.NpArangeNode, nm.NpLinspaceNode, nm.NpZerosNode, nm.NpOnesNode,
                    nm.NpRandNode, nm.NpRandnNode, nm.NpEyeNode,
                    nm.NpReduceNode, nm.NpArrayFuncNode,
                    nm.NpClipNode, nm.NpReshapeNode,
                    nm.NpDotNode, nm.NpMatMulNode, nm.NpInvNode]:
        g = Graph()
        g.add_node(NodeCls())
        script = _export(g)
        assert "export not" not in script, f"{NodeCls.__name__} unsupported"
    assert "import numpy as np" in _export(_build_single(nm.NpArangeNode))


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
    for NodeCls in [sk.SkTrainTestSplitNode, sk.SkScalerNode,
                    sk.SkEncoderNode, sk.SkLinearRegressionNode,
                    sk.SkClassifierNode, sk.SkPredictNode, sk.SkAccuracyNode,
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
                    pt.LossFnNode]:
        g = Graph()
        g.add_node(NodeCls())
        script = _export(g)
        assert "import torch" in script, f"{NodeCls.__name__} missing torch import"


def test_scheduler_nodes_export():
    import nodes.pytorch as pt
    for NodeCls in [pt.LRSchedulerNode]:
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
    g = Graph()
    a = _const(1.0); b = _const(2.0); add = _py("result = a + b")
    g.add_node(a); g.add_node(b); g.add_node(add)
    g.add_connection(a.id, "result", add.id, "a")
    g.add_connection(b.id, "result", add.id, "b")
    script = _export(g)
    ast.parse(script)
