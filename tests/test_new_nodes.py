"""Tests for tensor ops, recurrent, and backbone nodes."""
import pytest
import torch


# ── Tensor ops ────────────────────────────────────────────────────────────────

def test_tensor_cat():
    from nodes.pytorch.tensor_ops import TensorCatNode
    t1 = torch.ones(2, 3)
    t2 = torch.ones(2, 4)
    result = TensorCatNode().execute({"t1": t1, "t2": t2, "t3": None, "t4": None, "dim": 1})
    assert result["tensor"].shape == (2, 7)

def test_tensor_cat_none_guard():
    from nodes.pytorch.tensor_ops import TensorCatNode
    result = TensorCatNode().execute({"t1": None, "t2": None, "t3": None, "t4": None, "dim": 0})
    assert result["tensor"] is None

def test_tensor_stack():
    from nodes.pytorch.tensor_ops import TensorStackNode
    t1 = torch.ones(3)
    t2 = torch.ones(3)
    result = TensorStackNode().execute({"t1": t1, "t2": t2, "t3": None, "t4": None, "dim": 0})
    assert result["tensor"].shape == (2, 3)

def test_tensor_split():
    from nodes.pytorch.tensor_ops import TensorSplitNode
    t = torch.ones(6, 4)
    result = TensorSplitNode().execute({"tensor": t, "split_size": 3, "dim": 0})
    assert result["chunk_0"].shape == (3, 4)
    assert result["chunk_1"].shape == (3, 4)

def test_tensor_reshape():
    from nodes.pytorch.tensor_ops import TensorReshapeNode
    t = torch.ones(2, 3)
    result = TensorReshapeNode().execute({"tensor": t, "shape": "6"})
    assert result["tensor"].shape == (6,)

def test_tensor_reshape_infer():
    from nodes.pytorch.tensor_ops import TensorReshapeNode
    t = torch.ones(2, 3)
    result = TensorReshapeNode().execute({"tensor": t, "shape": "3,-1"})
    assert result["tensor"].shape == (3, 2)

def test_tensor_unsqueeze():
    from nodes.pytorch.tensor_ops import TensorUnsqueezeNode
    t = torch.ones(4)
    result = TensorUnsqueezeNode().execute({"tensor": t, "dim": 0})
    assert result["tensor"].shape == (1, 4)

def test_tensor_squeeze():
    from nodes.pytorch.tensor_ops import TensorSqueezeNode
    t = torch.ones(1, 4, 1)
    result = TensorSqueezeNode().execute({"tensor": t, "dim": -1})
    assert result["tensor"].shape == (4,)

def test_tensor_transpose():
    from nodes.pytorch.tensor_ops import TensorTransposeNode
    t = torch.ones(2, 3)
    result = TensorTransposeNode().execute({"tensor": t, "dim0": 0, "dim1": 1})
    assert result["tensor"].shape == (3, 2)

def test_tensor_permute():
    from nodes.pytorch.tensor_ops import TensorPermuteNode
    t = torch.ones(2, 3, 4)
    result = TensorPermuteNode().execute({"tensor": t, "dims": "2,0,1"})
    assert result["tensor"].shape == (4, 2, 3)

def test_tensor_einsum():
    from nodes.pytorch.tensor_ops import TensorEinsumNode
    t1 = torch.ones(2, 3)
    t2 = torch.ones(3, 4)
    result = TensorEinsumNode().execute({"equation": "ij,jk->ik", "t1": t1, "t2": t2})
    assert result["tensor"].shape == (2, 4)

def test_tensor_none_guards():
    from nodes.pytorch.tensor_ops import (TensorReshapeNode, TensorUnsqueezeNode,
                                           TensorSqueezeNode, TensorTransposeNode,
                                           TensorPermuteNode, TensorEinsumNode)
    for NodeCls, kwargs in [
        (TensorReshapeNode,   {"tensor": None, "shape": "4"}),
        (TensorUnsqueezeNode, {"tensor": None, "dim": 0}),
        (TensorSqueezeNode,   {"tensor": None, "dim": 0}),
        (TensorTransposeNode, {"tensor": None, "dim0": 0, "dim1": 1}),
        (TensorPermuteNode,   {"tensor": None, "dims": "0,1"}),
        (TensorEinsumNode,    {"equation": "ij,jk->ik", "t1": None, "t2": None}),
    ]:
        result = NodeCls().execute(kwargs)
        assert result["tensor"] is None, f"{NodeCls.__name__} should return None"


# ── Recurrent ─────────────────────────────────────────────────────────────────

def test_rnn_layer_creates_module():
    from nodes.pytorch.recurrent import RNNLayerNode
    import torch.nn as nn
    result = RNNLayerNode().execute({"input_size": 16, "hidden_size": 32, "num_layers": 1,
                                      "nonlinearity": "tanh", "dropout": 0.0,
                                      "bidirectional": False, "batch_first": True})
    assert isinstance(result["module"], nn.RNN)
    assert result["module"].hidden_size == 32

def test_gru_layer_creates_module():
    from nodes.pytorch.recurrent import GRULayerNode
    import torch.nn as nn
    result = GRULayerNode().execute({"input_size": 16, "hidden_size": 32, "num_layers": 1,
                                      "dropout": 0.0, "bidirectional": False, "batch_first": True})
    assert isinstance(result["module"], nn.GRU)

def test_lstm_layer_creates_module():
    from nodes.pytorch.recurrent import LSTMLayerNode
    import torch.nn as nn
    result = LSTMLayerNode().execute({"input_size": 16, "hidden_size": 32, "num_layers": 1,
                                       "dropout": 0.0, "bidirectional": False, "batch_first": True})
    assert isinstance(result["module"], nn.LSTM)

def test_rnn_forward():
    from nodes.pytorch.recurrent import RNNLayerNode, RNNForwardNode
    rnn = RNNLayerNode().execute({"input_size": 8, "hidden_size": 16, "num_layers": 1,
                                   "nonlinearity": "tanh", "dropout": 0.0,
                                   "bidirectional": False, "batch_first": True})["module"]
    x = torch.randn(4, 10, 8)  # (batch=4, seq=10, input=8)
    result = RNNForwardNode().execute({"module": rnn, "x": x, "h0": None})
    assert result["output"].shape == (4, 10, 16)  # (batch, seq, hidden)
    assert result["hidden"] is not None

def test_lstm_forward():
    from nodes.pytorch.recurrent import LSTMLayerNode, LSTMForwardNode
    lstm = LSTMLayerNode().execute({"input_size": 8, "hidden_size": 16, "num_layers": 1,
                                     "dropout": 0.0, "bidirectional": False, "batch_first": True})["module"]
    x = torch.randn(4, 10, 8)
    result = LSTMForwardNode().execute({"module": lstm, "x": x, "h0": None, "c0": None})
    assert result["output"].shape == (4, 10, 16)
    assert result["hidden"] is not None
    assert result["cell"] is not None

def test_lstm_forward_with_hidden():
    from nodes.pytorch.recurrent import LSTMLayerNode, LSTMForwardNode
    lstm = LSTMLayerNode().execute({"input_size": 8, "hidden_size": 16, "num_layers": 1,
                                     "dropout": 0.0, "bidirectional": False, "batch_first": True})["module"]
    x  = torch.randn(2, 5, 8)
    h0 = torch.zeros(1, 2, 16)
    c0 = torch.zeros(1, 2, 16)
    result = LSTMForwardNode().execute({"module": lstm, "x": x, "h0": h0, "c0": c0})
    assert result["output"].shape == (2, 5, 16)

def test_rnn_none_guard():
    from nodes.pytorch.recurrent import RNNForwardNode
    result = RNNForwardNode().execute({"module": None, "x": None, "h0": None})
    assert result["output"] is None and result["hidden"] is None

def test_lstm_none_guard():
    from nodes.pytorch.recurrent import LSTMForwardNode
    result = LSTMForwardNode().execute({"module": None, "x": None, "h0": None, "c0": None})
    assert result["output"] is None

def test_bidirectional_gru():
    from nodes.pytorch.recurrent import GRULayerNode, RNNForwardNode
    gru = GRULayerNode().execute({"input_size": 8, "hidden_size": 16, "num_layers": 1,
                                   "dropout": 0.0, "bidirectional": True, "batch_first": True})["module"]
    x = torch.randn(2, 5, 8)
    result = RNNForwardNode().execute({"module": gru, "x": x, "h0": None})
    # bidirectional doubles output features
    assert result["output"].shape == (2, 5, 32)


# ── Backbones ─────────────────────────────────────────────────────────────────

torchvision = pytest.importorskip("torchvision")

def test_resnet18_no_pretrained():
    from nodes.pytorch.backbones import ResNet18Node
    import torch.nn as nn
    result = ResNet18Node().execute({"pretrained": False, "num_classes": 5})
    assert result["model"] is not None
    assert isinstance(result["model"].fc, nn.Linear)
    assert result["model"].fc.out_features == 5
    assert "ResNet-18" in result["info"]

def test_resnet50_no_pretrained():
    from nodes.pytorch.backbones import ResNet50Node
    result = ResNet50Node().execute({"pretrained": False, "num_classes": 3})
    assert result["model"] is not None
    assert result["model"].fc.out_features == 3

def test_mobilenet_no_pretrained():
    from nodes.pytorch.backbones import MobileNetV3Node
    result = MobileNetV3Node().execute({"pretrained": False, "num_classes": 4})
    assert result["model"] is not None

def test_freeze_backbone():
    from nodes.pytorch.backbones import ResNet18Node, FreezeBackboneNode
    model = ResNet18Node().execute({"pretrained": False, "num_classes": 10})["model"]
    result = FreezeBackboneNode().execute({"model": model, "freeze_all": True})
    frozen_model = result["model"]
    assert all(not p.requires_grad for p in frozen_model.parameters())
    assert "frozen=" in result["info"]

def test_unfreeze_backbone():
    from nodes.pytorch.backbones import ResNet18Node, FreezeBackboneNode
    model = ResNet18Node().execute({"pretrained": False, "num_classes": 10})["model"]
    # Freeze first
    FreezeBackboneNode().execute({"model": model, "freeze_all": True})
    # Unfreeze
    result = FreezeBackboneNode().execute({"model": model, "freeze_all": False})
    assert all(p.requires_grad for p in result["model"].parameters())

def test_model_info():
    from nodes.pytorch.backbones import ResNet18Node, ModelInfoNode
    model = ResNet18Node().execute({"pretrained": False, "num_classes": 10})["model"]
    result = ModelInfoNode().execute({"model": model})
    assert "trainable=" in result["info"]
    assert result["model"] is model  # passes through unchanged

def test_model_info_none_guard():
    from nodes.pytorch.backbones import ModelInfoNode
    result = ModelInfoNode().execute({"model": None})
    assert result["model"] is None


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_has_new_nodes():
    from nodes import NODE_REGISTRY
    expected = [
        "pt_tensor_cat", "pt_tensor_stack", "pt_tensor_split", "pt_tensor_reshape",
        "pt_tensor_unsqueeze", "pt_tensor_squeeze", "pt_tensor_transpose",
        "pt_tensor_permute", "pt_tensor_einsum",
        "pt_rnn_layer", "pt_gru_layer", "pt_lstm_layer",
        "pt_rnn_forward", "pt_lstm_forward",
        "pt_resnet18", "pt_resnet50", "pt_mobilenet_v3", "pt_efficientnet_b0",
        "pt_freeze_backbone", "pt_model_info",
    ]
    for tn in expected:
        assert tn in NODE_REGISTRY, f"Missing: {tn}"

def test_subcategories_set():
    from nodes import NODE_REGISTRY
    checks = {
        "pt_tensor_cat":   "Tensors",
        "pt_rnn_layer":    "Recurrent",
        "pt_lstm_forward": "Recurrent",
        "pt_resnet18":     "Pretrained",
        "pt_freeze_backbone": "Pretrained",
    }
    for tn, expected_sub in checks.items():
        cls = NODE_REGISTRY[tn]
        assert cls.subcategory == expected_sub, f"{tn}: expected '{expected_sub}', got '{cls.subcategory}'"
