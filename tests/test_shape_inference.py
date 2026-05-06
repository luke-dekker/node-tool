"""Shape inference across LayerNode kinds — Linear, Conv2d, BatchNorm*,
LayerNorm, LSTM, GRU, RNN, MultiheadAttention, TransformerEncoderLayer,
PositionalEncoding all infer their input dimension from the upstream
tensor instead of a hardcoded port. Width mutations propagate cleanly
through the chain without manual `in_features` / `in_channels` wiring.

Post-consolidation: LayerNode (kind dropdown) absorbs the simple wrappers,
RecurrentLayerNode absorbs RNN/LSTM/GRU. MultiheadAttentionNode stays
standalone (3 tensor inputs).
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _register_port_types():
    from plugins.agents.port_types import register_all
    register_all()


# ── Linear ────────────────────────────────────────────────────────────────

def _layer(kind: str, **inputs):
    """Helper: build a LayerNode pre-configured for a kind, return execute fn."""
    from nodes.pytorch.layer import LayerNode
    n = LayerNode()
    return n


def test_linear_infers_in_features_from_tensor_shape():
    from nodes.pytorch.layer import LayerNode
    n = LayerNode()
    x = torch.randn(4, 123)
    out = n.execute({"tensor_in": x, "kind": "linear", "out_features": 16, "bias": True})
    assert out["tensor_out"].shape == (4, 16)
    assert n._layer.in_features == 123   # inferred, not the default 64


def test_linear_rebuilds_when_upstream_width_changes():
    from nodes.pytorch.layer import LayerNode
    n = LayerNode()
    n.execute({"tensor_in": torch.randn(2, 64), "kind": "linear", "out_features": 10})
    first_layer = n._layer
    assert first_layer.in_features == 64
    n.execute({"tensor_in": torch.randn(2, 128), "kind": "linear", "out_features": 10})
    assert n._layer.in_features == 128
    assert n._layer is not first_layer


# ── Conv2d ────────────────────────────────────────────────────────────────

def test_conv2d_infers_in_channels():
    from nodes.pytorch.layer import LayerNode
    n = LayerNode()
    x = torch.randn(2, 7, 32, 32)
    out = n.execute({"tensor_in": x, "kind": "conv2d", "out_ch": 16, "kernel": 3, "padding": 1})
    assert out["tensor_out"] is not None
    assert out["tensor_out"].shape == (2, 16, 32, 32)
    assert n._layer.in_channels == 7


# ── BatchNorm ─────────────────────────────────────────────────────────────

def test_batchnorm1d_infers_num_features():
    from nodes.pytorch.layer import LayerNode
    n = LayerNode()
    x = torch.randn(4, 20)
    out = n.execute({"tensor_in": x, "kind": "batchnorm1d"})
    assert out["tensor_out"].shape == (4, 20)
    assert n._layer.num_features == 20


def test_batchnorm2d_infers_num_features():
    from nodes.pytorch.layer import LayerNode
    n = LayerNode()
    x = torch.randn(2, 5, 16, 16)
    out = n.execute({"tensor_in": x, "kind": "batchnorm2d"})
    assert out["tensor_out"].shape == (2, 5, 16, 16)
    assert n._layer.num_features == 5


# ── LayerNorm ─────────────────────────────────────────────────────────────

def test_layernorm_infers_normalized_shape():
    from nodes.pytorch.layer import LayerNode
    n = LayerNode()
    x = torch.randn(4, 10, 77)
    out = n.execute({"tensor_in": x, "kind": "layernorm", "eps": 1e-5})
    assert out["tensor_out"].shape == (4, 10, 77)
    assert n._layer.normalized_shape == (77,)


# ── RNN family (RecurrentLayerNode) ───────────────────────────────────────

def test_lstm_infers_input_size():
    from nodes.pytorch.recurrent_layer import RecurrentLayerNode
    n = RecurrentLayerNode()
    x = torch.randn(2, 5, 48)
    out = n.execute({"input_seq": x, "kind": "lstm", "hidden_size": 16, "num_layers": 1,
                     "dropout": 0.0, "bidirectional": False, "batch_first": True})
    assert out["output"].shape == (2, 5, 16)
    assert n._layer.input_size == 48


def test_gru_infers_input_size():
    from nodes.pytorch.recurrent_layer import RecurrentLayerNode
    n = RecurrentLayerNode()
    x = torch.randn(2, 5, 48)
    out = n.execute({"input_seq": x, "kind": "gru", "hidden_size": 16, "num_layers": 1,
                     "dropout": 0.0, "bidirectional": False, "batch_first": True})
    assert out["output"].shape == (2, 5, 16)
    assert n._layer.input_size == 48


def test_rnn_infers_input_size():
    from nodes.pytorch.recurrent_layer import RecurrentLayerNode
    n = RecurrentLayerNode()
    x = torch.randn(2, 5, 48)
    out = n.execute({"input_seq": x, "kind": "rnn", "hidden_size": 16, "num_layers": 1,
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
    from nodes.pytorch.layer import LayerNode
    n = LayerNode()
    x = torch.randn(2, 5, 64)
    out = n.execute({"tensor_in": x, "kind": "transformer_encoder",
                     "nhead": 8, "dim_feedforward": 128,
                     "dropout": 0.0, "activation": "relu"})
    assert out["tensor_out"].shape == (2, 5, 64)


def test_positional_encoding_infers_d_model():
    from nodes.pytorch.layer import LayerNode
    n = LayerNode()
    x = torch.randn(2, 5, 32)
    out = n.execute({"tensor_in": x, "kind": "positional_encoding",
                     "max_len": 100, "pe_kind": "sinusoidal"})
    assert out["tensor_out"].shape == (2, 5, 32)


# ── End-to-end: width mutation propagates through a two-layer chain ───────

def test_width_mutation_propagates_linear_then_bn():
    """Linear → BatchNorm1d → Linear. Mutating the first Linear's
    out_features should cascade: BN rebuilds with new num_features, the
    second Linear rebuilds with new in_features."""
    from nodes.pytorch.layer import LayerNode

    l1 = LayerNode(); bn = LayerNode(); l2 = LayerNode()

    x = torch.randn(4, 20)
    h1 = l1.execute({"tensor_in": x, "kind": "linear", "out_features": 8})["tensor_out"]
    hb = bn.execute({"tensor_in": h1, "kind": "batchnorm1d"})["tensor_out"]
    h2 = l2.execute({"tensor_in": hb, "kind": "linear", "out_features": 2})["tensor_out"]
    assert h2.shape == (4, 2)
    assert bn._layer.num_features == 8
    assert l2._layer.in_features == 8

    h1 = l1.execute({"tensor_in": x, "kind": "linear", "out_features": 16})["tensor_out"]
    hb = bn.execute({"tensor_in": h1, "kind": "batchnorm1d"})["tensor_out"]
    h2 = l2.execute({"tensor_in": hb, "kind": "linear", "out_features": 2})["tensor_out"]
    assert h2.shape == (4, 2)
    assert l1._layer.out_features == 16
    assert bn._layer.num_features == 16
    assert l2._layer.in_features == 16
