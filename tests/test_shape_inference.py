"""Shape inference across layer nodes — Linear, Conv2d, BatchNorm*,
LayerNorm, LSTM, GRU, RNN, MultiheadAttention, TransformerEncoderLayer,
PositionalEncoding all infer their input dimension from the upstream
tensor instead of a hardcoded port. Width mutations propagate cleanly
through the chain without manual `in_features` / `in_channels` wiring.
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _register_port_types():
    from plugins.agents.port_types import register_all
    register_all()


# ── Linear ────────────────────────────────────────────────────────────────

def test_linear_infers_in_features_from_tensor_shape():
    from nodes.pytorch.linear import LinearNode
    n = LinearNode()
    x = torch.randn(4, 123)
    out = n.execute({"tensor_in": x, "out_features": 16, "bias": True})
    assert out["tensor_out"].shape == (4, 16)
    assert n._layer.in_features == 123   # inferred, not the default 64


def test_linear_rebuilds_when_upstream_width_changes():
    from nodes.pytorch.linear import LinearNode
    n = LinearNode()
    n.execute({"tensor_in": torch.randn(2, 64), "out_features": 10})
    first_layer = n._layer
    assert first_layer.in_features == 64
    # Upstream width doubles — layer should rebuild.
    n.execute({"tensor_in": torch.randn(2, 128), "out_features": 10})
    assert n._layer.in_features == 128
    assert n._layer is not first_layer


# ── Conv2d ────────────────────────────────────────────────────────────────

def test_conv2d_infers_in_channels():
    from nodes.pytorch.conv2d import Conv2dNode
    n = Conv2dNode()
    # NCHW: channel dim is axis=1
    x = torch.randn(2, 7, 32, 32)
    out = n.execute({"tensor_in": x, "out_ch": 16, "kernel": 3, "padding": 1})
    assert out["tensor_out"] is not None
    assert out["tensor_out"].shape == (2, 16, 32, 32)
    assert n._layer.in_channels == 7


# ── BatchNorm ─────────────────────────────────────────────────────────────

def test_batchnorm1d_infers_num_features():
    from nodes.pytorch.batchnorm1d import BatchNorm1dNode
    n = BatchNorm1dNode()
    x = torch.randn(4, 20)
    out = n.execute({"tensor_in": x})
    assert out["tensor_out"].shape == (4, 20)
    assert n._layer.num_features == 20


def test_batchnorm2d_infers_num_features():
    from nodes.pytorch.batchnorm2d import BatchNorm2dNode
    n = BatchNorm2dNode()
    x = torch.randn(2, 5, 16, 16)
    out = n.execute({"tensor_in": x})
    assert out["tensor_out"].shape == (2, 5, 16, 16)
    assert n._layer.num_features == 5


# ── LayerNorm ─────────────────────────────────────────────────────────────

def test_layernorm_infers_normalized_shape():
    from nodes.pytorch.layer_norm import LayerNormNode
    n = LayerNormNode()
    x = torch.randn(4, 10, 77)
    out = n.execute({"tensor_in": x, "eps": 1e-5})
    assert out["tensor_out"].shape == (4, 10, 77)
    # LayerNorm's normalized_shape is a tuple
    assert n._layer.normalized_shape == (77,)


# ── RNN family ────────────────────────────────────────────────────────────

def test_lstm_infers_input_size():
    from nodes.pytorch.lstm import LSTMNode
    n = LSTMNode()
    x = torch.randn(2, 5, 48)   # (B, T, features)
    out = n.execute({"x": x, "hidden_size": 16, "num_layers": 1,
                      "dropout": 0.0, "bidirectional": False,
                      "batch_first": True})
    assert out["output"].shape == (2, 5, 16)
    assert n._layer.input_size == 48


def test_gru_infers_input_size():
    from nodes.pytorch.gru import GRUNode
    n = GRUNode()
    x = torch.randn(2, 5, 48)
    out = n.execute({"x": x, "hidden_size": 16, "num_layers": 1,
                      "dropout": 0.0, "bidirectional": False,
                      "batch_first": True})
    assert out["output"].shape == (2, 5, 16)
    assert n._layer.input_size == 48


def test_rnn_infers_input_size():
    from nodes.pytorch.rnn import RNNNode
    n = RNNNode()
    x = torch.randn(2, 5, 48)
    out = n.execute({"x": x, "hidden_size": 16, "num_layers": 1,
                      "nonlinearity": "tanh", "dropout": 0.0,
                      "bidirectional": False, "batch_first": True})
    assert out["output"].shape == (2, 5, 16)
    assert n._layer.input_size == 48


# ── Attention / Transformer / Positional ──────────────────────────────────

def test_multihead_attention_infers_embed_dim():
    from nodes.pytorch.multihead_attention import MultiheadAttentionNode
    n = MultiheadAttentionNode()
    q = torch.randn(2, 5, 64)
    out = n.execute({"query": q, "key": None, "value": None,
                      "num_heads": 8, "dropout": 0.0})
    assert out["tensor_out"].shape == (2, 5, 64)
    assert n._layer.embed_dim == 64


def test_transformer_encoder_layer_infers_d_model():
    from nodes.pytorch.transformer_encoder_layer import TransformerEncoderLayerNode
    n = TransformerEncoderLayerNode()
    x = torch.randn(2, 5, 64)
    out = n.execute({"tensor_in": x, "nhead": 8, "dim_feedforward": 128,
                      "dropout": 0.0, "activation": "relu"})
    assert out["tensor_out"].shape == (2, 5, 64)


def test_positional_encoding_infers_d_model():
    from nodes.pytorch.positional_encoding import PositionalEncodingNode
    n = PositionalEncodingNode()
    x = torch.randn(2, 5, 32)
    out = n.execute({"tensor_in": x, "max_len": 100, "kind": "sinusoidal"})
    assert out["tensor_out"].shape == (2, 5, 32)


# ── End-to-end: width mutation propagates through a two-layer chain ───────

def test_width_mutation_propagates_linear_then_bn():
    """Linear → BatchNorm1d → Linear. Mutating the first Linear's
    out_features should cascade: BN rebuilds with new num_features, the
    second Linear rebuilds with new in_features."""
    from nodes.pytorch.linear import LinearNode
    from nodes.pytorch.batchnorm1d import BatchNorm1dNode

    l1 = LinearNode()
    bn = BatchNorm1dNode()
    l2 = LinearNode()

    x = torch.randn(4, 20)
    # Round 1: l1 width = 8
    h1 = l1.execute({"tensor_in": x, "out_features": 8})["tensor_out"]
    hb = bn.execute({"tensor_in": h1})["tensor_out"]
    h2 = l2.execute({"tensor_in": hb, "out_features": 2})["tensor_out"]
    assert h2.shape == (4, 2)
    assert bn._layer.num_features == 8
    assert l2._layer.in_features == 8

    # Round 2: l1 width = 16 — everything downstream should reshape.
    h1 = l1.execute({"tensor_in": x, "out_features": 16})["tensor_out"]
    hb = bn.execute({"tensor_in": h1})["tensor_out"]
    h2 = l2.execute({"tensor_in": hb, "out_features": 2})["tensor_out"]
    assert h2.shape == (4, 2)
    assert l1._layer.out_features == 16
    assert bn._layer.num_features == 16
    assert l2._layer.in_features == 16
