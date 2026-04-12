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
    from nodes.pytorch.tensor_shape_op import TensorShapeOpNode
    t = torch.ones(2, 3)
    result = TensorShapeOpNode().execute({"mode": "reshape", "tensor": t, "shape": "3,-1"})
    assert result["tensor"].shape == (3, 2)

def test_tensor_unsqueeze():
    from nodes.pytorch.tensor_shape_op import TensorShapeOpNode
    t = torch.ones(4)
    result = TensorShapeOpNode().execute({"mode": "unsqueeze", "tensor": t, "dim": 0})
    assert result["tensor"].shape == (1, 4)

def test_tensor_squeeze():
    from nodes.pytorch.tensor_shape_op import TensorShapeOpNode
    t = torch.ones(1, 4, 1)
    result = TensorShapeOpNode().execute({"mode": "squeeze", "tensor": t, "dim": -1})
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
    from nodes.pytorch.tensor_shape_op import TensorShapeOpNode
    from nodes.pytorch.tensor_ops import (TensorTransposeNode,
                                           TensorPermuteNode, TensorEinsumNode)
    for NodeCls, kwargs in [
        (TensorShapeOpNode,   {"mode": "reshape", "tensor": None, "shape": "4"}),
        (TensorShapeOpNode,   {"mode": "unsqueeze", "tensor": None, "dim": 0}),
        (TensorShapeOpNode,   {"mode": "squeeze", "tensor": None, "dim": 0}),
        (TensorTransposeNode, {"tensor": None, "dim0": 0, "dim1": 1}),
        (TensorPermuteNode,   {"tensor": None, "dims": "0,1"}),
        (TensorEinsumNode,    {"equation": "ij,jk->ik", "t1": None, "t2": None}),
    ]:
        result = NodeCls().execute(kwargs)
        assert result["tensor"] is None, f"{NodeCls.__name__} should return None"


# ── Recurrent ─────────────────────────────────────────────────────────────────

def test_rnn_creates_module():
    from nodes.pytorch.rnn import RNNNode
    import torch.nn as nn
    result = RNNNode().execute({"input_size": 16, "hidden_size": 32, "num_layers": 1,
                                 "nonlinearity": "tanh", "dropout": 0.0,
                                 "bidirectional": False, "batch_first": True})
    assert isinstance(result["module"], nn.RNN)
    assert result["module"].hidden_size == 32

def test_gru_creates_module():
    from nodes.pytorch.gru import GRUNode
    import torch.nn as nn
    result = GRUNode().execute({"input_size": 16, "hidden_size": 32, "num_layers": 1,
                                 "dropout": 0.0, "bidirectional": False, "batch_first": True})
    assert isinstance(result["module"], nn.GRU)

def test_lstm_creates_module():
    from nodes.pytorch.lstm import LSTMNode
    import torch.nn as nn
    result = LSTMNode().execute({"input_size": 16, "hidden_size": 32, "num_layers": 1,
                                  "dropout": 0.0, "bidirectional": False, "batch_first": True})
    assert isinstance(result["module"], nn.LSTM)

def test_rnn_forward():
    from nodes.pytorch.rnn import RNNNode
    x = torch.randn(4, 10, 8)  # (batch=4, seq=10, input=8)
    result = RNNNode().execute({"input_size": 8, "hidden_size": 16, "num_layers": 1,
                                 "nonlinearity": "tanh", "dropout": 0.0,
                                 "bidirectional": False, "batch_first": True,
                                 "x": x, "h0": None})
    assert result["output"].shape == (4, 10, 16)
    assert result["hidden"] is not None

def test_lstm_forward():
    from nodes.pytorch.lstm import LSTMNode
    x = torch.randn(4, 10, 8)
    result = LSTMNode().execute({"input_size": 8, "hidden_size": 16, "num_layers": 1,
                                  "dropout": 0.0, "bidirectional": False, "batch_first": True,
                                  "x": x, "h0": None, "c0": None})
    assert result["output"].shape == (4, 10, 16)
    assert result["hidden"] is not None
    assert result["cell"] is not None

def test_lstm_forward_with_hidden():
    from nodes.pytorch.lstm import LSTMNode
    x  = torch.randn(2, 5, 8)
    h0 = torch.zeros(1, 2, 16)
    c0 = torch.zeros(1, 2, 16)
    result = LSTMNode().execute({"input_size": 8, "hidden_size": 16, "num_layers": 1,
                                  "dropout": 0.0, "bidirectional": False, "batch_first": True,
                                  "x": x, "h0": h0, "c0": c0})
    assert result["output"].shape == (2, 5, 16)

def test_rnn_none_guard():
    from nodes.pytorch.rnn import RNNNode
    result = RNNNode().execute({"x": None, "h0": None})
    assert result["output"] is None and result["hidden"] is None

def test_lstm_none_guard():
    from nodes.pytorch.lstm import LSTMNode
    result = LSTMNode().execute({"x": None, "h0": None, "c0": None})
    assert result["output"] is None

def test_bidirectional_gru():
    from nodes.pytorch.gru import GRUNode
    x = torch.randn(2, 5, 8)
    result = GRUNode().execute({"input_size": 8, "hidden_size": 16, "num_layers": 1,
                                 "dropout": 0.0, "bidirectional": True, "batch_first": True,
                                 "x": x, "h0": None})
    # bidirectional doubles output features
    assert result["output"].shape == (2, 5, 32)


# ── Backbones ─────────────────────────────────────────────────────────────────

torchvision = pytest.importorskip("torchvision")

def test_resnet18_no_pretrained():
    from nodes.pytorch.pretrained_backbone import PretrainedBackboneNode
    import torch.nn as nn
    result = PretrainedBackboneNode().execute({"architecture": "resnet18", "pretrained": False, "num_classes": 5})
    assert result["model"] is not None
    assert isinstance(result["model"].fc, nn.Linear)
    assert result["model"].fc.out_features == 5
    assert "ResNet-18" in result["info"]

def test_resnet50_no_pretrained():
    from nodes.pytorch.pretrained_backbone import PretrainedBackboneNode
    result = PretrainedBackboneNode().execute({"architecture": "resnet50", "pretrained": False, "num_classes": 3})
    assert result["model"] is not None
    assert result["model"].fc.out_features == 3

def test_mobilenet_no_pretrained():
    from nodes.pytorch.pretrained_backbone import PretrainedBackboneNode
    result = PretrainedBackboneNode().execute({"architecture": "mobilenet_v3_small", "pretrained": False, "num_classes": 4})
    assert result["model"] is not None

def test_freeze_backbone():
    from nodes.pytorch.pretrained_backbone import PretrainedBackboneNode
    from nodes.pytorch.freeze_backbone import FreezeBackboneNode
    model = PretrainedBackboneNode().execute({"architecture": "resnet18", "pretrained": False, "num_classes": 10})["model"]
    result = FreezeBackboneNode().execute({"model": model, "freeze_all": True})
    frozen_model = result["model"]
    assert all(not p.requires_grad for p in frozen_model.parameters())
    assert "frozen=" in result["info"]

def test_unfreeze_backbone():
    from nodes.pytorch.pretrained_backbone import PretrainedBackboneNode
    from nodes.pytorch.freeze_backbone import FreezeBackboneNode
    model = PretrainedBackboneNode().execute({"architecture": "resnet18", "pretrained": False, "num_classes": 10})["model"]
    FreezeBackboneNode().execute({"model": model, "freeze_all": True})
    result = FreezeBackboneNode().execute({"model": model, "freeze_all": False})
    assert all(p.requires_grad for p in result["model"].parameters())

def test_model_info():
    from nodes.pytorch.pretrained_backbone import PretrainedBackboneNode
    from nodes.pytorch.model_info_persist import ModelInfoNode
    model = PretrainedBackboneNode().execute({"architecture": "resnet18", "pretrained": False, "num_classes": 10})["model"]
    result = ModelInfoNode().execute({"model": model})
    assert "Trainable params:" in result["info"]
    assert result["model"] is model

def test_model_info_none_guard():
    from nodes.pytorch.backbones import ModelInfoNode
    result = ModelInfoNode().execute({"model": None})
    assert result["model"] is None


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_has_new_nodes():
    from nodes import NODE_REGISTRY
    expected = [
        "pt_tensor_cat", "pt_tensor_stack", "pt_tensor_split",
        "pt_tensor_shape_op", "pt_tensor_create",
        "pt_tensor_transpose", "pt_tensor_permute", "pt_tensor_einsum",
        "pt_rnn", "pt_lstm", "pt_gru",
        "pt_pretrained_backbone",
        "pt_freeze_backbone", "pt_model_info",
    ]
    for tn in expected:
        assert tn in NODE_REGISTRY, f"Missing: {tn}"

def test_subcategories_set():
    from nodes import NODE_REGISTRY
    checks = {
        "pt_tensor_cat":   "",
        "pt_rnn":          "Recurrent",
        "pt_lstm":         "Recurrent",
        "pt_pretrained_backbone": "Pretrained",
        "pt_freeze_backbone": "Pretrained",
    }
    for tn, expected_sub in checks.items():
        cls = NODE_REGISTRY[tn]
        assert cls.subcategory == expected_sub, f"{tn}: expected '{expected_sub}', got '{cls.subcategory}'"
