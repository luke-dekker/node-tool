"""Tests for layer nodes — tensor forward-pass and persistent state.

Post-consolidation: per-class layer nodes (LinearNode, Conv2dNode, etc.) all
collapse into LayerNode (kind dropdown). Tests build a LayerNode and set
`kind` to dispatch to the right code path.
"""
import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn

from nodes.pytorch.layer   import LayerNode
from nodes.pytorch.flatten import FlattenNode


def _exec(node, **kw):
    return node.execute(kw)


def _layer(kind: str) -> LayerNode:
    """Build a LayerNode pre-configured for `kind`."""
    n = LayerNode()
    n.inputs["kind"].default_value = kind
    return n


# ── Tensor forward-pass ───────────────────────────────────────────────────────

def test_flatten_output_shape():
    out = _exec(FlattenNode(), tensor_in=torch.randn(1, 1, 28, 28), start_dim=1)["tensor_out"]
    assert out.shape == (1, 784)

def test_flatten_none_input_returns_none():
    out = _exec(FlattenNode(), tensor_in=None, start_dim=1)["tensor_out"]
    assert out is None

def test_linear_output_shape():
    out = _exec(_layer("linear"), tensor_in=torch.randn(1, 784),
                kind="linear", out_features=256, bias=True,
                activation="none", freeze=False)["tensor_out"]
    assert out.shape == (1, 256)

def test_linear_relu_activation():
    out = _exec(_layer("linear"), tensor_in=torch.randn(2, 4),
                kind="linear", out_features=8, bias=True,
                activation="relu", freeze=False)["tensor_out"]
    assert out.shape == (2, 8)
    assert (out >= 0).all()

def test_linear_sigmoid_activation():
    out = _exec(_layer("linear"), tensor_in=torch.randn(2, 4),
                kind="linear", out_features=4, bias=False,
                activation="sigmoid", freeze=False)["tensor_out"]
    assert (out > 0).all() and (out < 1).all()

def test_linear_none_input_returns_none():
    # With placeholder in_f=1 during priming, the layer materializes but
    # the forward is skipped (no input tensor) — returns None.
    out = _exec(_layer("linear"), tensor_in=None,
                kind="linear", out_features=8, bias=True,
                activation="none", freeze=False)["tensor_out"]
    assert out is None

def test_full_tensor_pipeline():
    flat = FlattenNode()
    lin1 = _layer("linear")
    lin2 = _layer("linear")
    x = torch.randn(1, 1, 28, 28)
    t1 = _exec(flat, tensor_in=x, start_dim=1)["tensor_out"]
    assert t1.shape == (1, 784)
    t2 = _exec(lin1, tensor_in=t1, kind="linear", out_features=256,
               bias=True, activation="relu", freeze=False)["tensor_out"]
    assert t2.shape == (1, 256)
    t3 = _exec(lin2, tensor_in=t2, kind="linear", out_features=10,
               bias=True, activation="none", freeze=False)["tensor_out"]
    assert t3.shape == (1, 10)

def test_conv2d_output_shape():
    out = _exec(_layer("conv2d"), tensor_in=torch.randn(1, 1, 8, 8),
                kind="conv2d", out_ch=4, kernel=3, stride=1, padding=0,
                activation="none", freeze=False)["tensor_out"]
    assert out.shape == (1, 4, 6, 6)

def test_maxpool_output_shape():
    out = _exec(_layer("max_pool2d"), tensor_in=torch.randn(1, 1, 8, 8),
                kind="max_pool2d", kernel=2, stride=2)["tensor_out"]
    assert out.shape == (1, 1, 4, 4)

def test_avgpool_output_shape():
    out = _exec(_layer("avg_pool2d"), tensor_in=torch.randn(1, 1, 8, 8),
                kind="avg_pool2d", kernel=2, stride=2)["tensor_out"]
    assert out.shape == (1, 1, 4, 4)

def test_activation_relu():
    out = _exec(_layer("activation"), tensor_in=torch.randn(2, 4),
                kind="activation", activation="relu")["tensor_out"]
    assert (out >= 0).all()

def test_activation_none_passthrough():
    # "activation=none" produces nn.Identity(); the output is a copy, not the
    # exact same tensor object — but value-wise unchanged.
    x = torch.randn(2, 4)
    out = _exec(_layer("activation"), tensor_in=x,
                kind="activation", activation="none")["tensor_out"]
    assert torch.equal(out, x)

def test_dropout_shape_preserved():
    out = _exec(_layer("dropout"), tensor_in=torch.ones(2, 8),
                kind="dropout", p=0.5)["tensor_out"]
    assert out.shape == (2, 8)
