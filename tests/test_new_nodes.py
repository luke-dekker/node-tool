"""Tests for tensor ops, recurrent, and backbone nodes."""
import pytest
import torch


# ── Tensor ops ────────────────────────────────────────────────────────────────

def _reshape():
    from nodes.pytorch.tensor_reshape import TensorReshapeNode
    return TensorReshapeNode()

def _op():
    from nodes.pytorch.tensor_op import TensorOpNode
    return TensorOpNode()

def test_tensor_cat():
    t1 = torch.ones(2, 3); t2 = torch.ones(2, 4)
    result = _reshape().execute({"op": "cat", "t1": t1, "t2": t2, "dim": 1})
    assert result["tensor"].shape == (2, 7)

def test_tensor_cat_none_guard():
    result = _reshape().execute({"op": "cat", "t1": None, "t2": None, "dim": 0})
    assert result["tensor"] is None

def test_tensor_stack():
    t1 = torch.ones(3); t2 = torch.ones(3)
    result = _reshape().execute({"op": "stack", "t1": t1, "t2": t2, "dim": 0})
    assert result["tensor"].shape == (2, 3)

def test_tensor_split():
    t = torch.ones(6, 4)
    result = _reshape().execute({"op": "split", "t1": t, "split_size": 3, "dim": 0})
    assert result["chunk_0"].shape == (3, 4)
    assert result["chunk_1"].shape == (3, 4)

def test_tensor_reshape():
    t = torch.ones(2, 3)
    result = _reshape().execute({"op": "reshape", "t1": t, "shape": "6"})
    assert result["tensor"].shape == (6,)

def test_tensor_reshape_infer():
    t = torch.ones(2, 3)
    result = _reshape().execute({"op": "reshape", "t1": t, "shape": "3,-1"})
    assert result["tensor"].shape == (3, 2)

def test_tensor_unsqueeze():
    t = torch.ones(4)
    result = _reshape().execute({"op": "unsqueeze", "t1": t, "dim": 0})
    assert result["tensor"].shape == (1, 4)

def test_tensor_squeeze():
    t = torch.ones(1, 4, 1)
    result = _reshape().execute({"op": "squeeze", "t1": t, "dim": -1})
    assert result["tensor"].shape == (4,)

def test_tensor_transpose():
    t = torch.ones(2, 3)
    result = _reshape().execute({"op": "transpose", "t1": t, "dim": 0, "dim_b": 1})
    assert result["tensor"].shape == (3, 2)

def test_tensor_permute():
    t = torch.ones(2, 3, 4)
    result = _reshape().execute({"op": "permute", "t1": t, "shape": "2,0,1"})
    assert result["tensor"].shape == (4, 2, 3)

def test_tensor_einsum():
    t1 = torch.ones(2, 3); t2 = torch.ones(3, 4)
    result = _op().execute({"op": "einsum", "equation": "ij,jk->ik", "a": t1, "b": t2})
    assert result["result"].shape == (2, 4)

def test_tensor_none_guards():
    for kwargs in [
        {"op": "reshape",   "t1": None, "shape": "4"},
        {"op": "unsqueeze", "t1": None, "dim": 0},
        {"op": "squeeze",   "t1": None, "dim": 0},
        {"op": "transpose", "t1": None, "dim": 0, "dim_b": 1},
        {"op": "permute",   "t1": None, "shape": "0,1"},
    ]:
        result = _reshape().execute(kwargs)
        assert result["tensor"] is None, f"{kwargs['op']} should return None"
    result = _op().execute({"op": "einsum", "equation": "ij,jk->ik", "a": None, "b": None})
    assert result["result"] is None


# ── Recurrent ─────────────────────────────────────────────────────────────────

def _recurrent_node():
    """RNN/LSTM/GRU all live in RecurrentLayerNode (kind dropdown)."""
    from nodes.pytorch.recurrent_layer import RecurrentLayerNode
    return RecurrentLayerNode()

def test_rnn_creates_module():
    import torch.nn as nn
    result = _recurrent_node().execute({"kind": "rnn", "input_seq": torch.randn(1, 1, 16),
                                         "hidden_size": 32, "num_layers": 1,
                                         "nonlinearity": "tanh", "dropout": 0.0,
                                         "bidirectional": False, "batch_first": True})
    assert isinstance(result["module"], nn.RNN)
    assert result["module"].hidden_size == 32

def test_gru_creates_module():
    import torch.nn as nn
    result = _recurrent_node().execute({"kind": "gru", "input_seq": torch.randn(1, 1, 16),
                                         "hidden_size": 32, "num_layers": 1,
                                         "dropout": 0.0, "bidirectional": False, "batch_first": True})
    assert isinstance(result["module"], nn.GRU)

def test_lstm_creates_module():
    import torch.nn as nn
    result = _recurrent_node().execute({"kind": "lstm", "input_seq": torch.randn(1, 1, 16),
                                         "hidden_size": 32, "num_layers": 1,
                                         "dropout": 0.0, "bidirectional": False, "batch_first": True})
    assert isinstance(result["module"], nn.LSTM)

def test_rnn_forward():
    x = torch.randn(4, 10, 8)
    result = _recurrent_node().execute({"kind": "rnn", "input_seq": x,
                                         "hidden_size": 16, "num_layers": 1,
                                         "nonlinearity": "tanh", "dropout": 0.0,
                                         "bidirectional": False, "batch_first": True,
                                         "init_hidden": None})
    assert result["output"].shape == (4, 10, 16)
    assert result["hidden"] is not None

def test_lstm_forward():
    x = torch.randn(4, 10, 8)
    result = _recurrent_node().execute({"kind": "lstm", "input_seq": x,
                                         "hidden_size": 16, "num_layers": 1,
                                         "dropout": 0.0, "bidirectional": False, "batch_first": True,
                                         "init_hidden": None, "init_cell": None})
    assert result["output"].shape == (4, 10, 16)
    assert result["hidden"] is not None
    assert result["cell"] is not None

def test_lstm_forward_with_hidden():
    x  = torch.randn(2, 5, 8)
    h0 = torch.zeros(1, 2, 16)
    c0 = torch.zeros(1, 2, 16)
    result = _recurrent_node().execute({"kind": "lstm", "input_seq": x,
                                         "hidden_size": 16, "num_layers": 1,
                                         "dropout": 0.0, "bidirectional": False, "batch_first": True,
                                         "init_hidden": h0, "init_cell": c0})
    assert result["output"].shape == (2, 5, 16)

def test_rnn_none_guard():
    result = _recurrent_node().execute({"kind": "rnn", "input_seq": None, "init_hidden": None})
    assert result["output"] is None and result["hidden"] is None

def test_lstm_none_guard():
    result = _recurrent_node().execute({"kind": "lstm", "input_seq": None,
                                         "init_hidden": None, "init_cell": None})
    assert result["output"] is None

def test_bidirectional_gru():
    x = torch.randn(2, 5, 8)
    result = _recurrent_node().execute({"kind": "gru", "input_seq": x,
                                         "hidden_size": 16, "num_layers": 1,
                                         "dropout": 0.0, "bidirectional": True, "batch_first": True,
                                         "init_hidden": None})
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
    # 13 per-op tensor nodes collapsed into pt_tensor_op + pt_tensor_reshape
    # (plus pt_tensor_create/info/print). RNN/LSTM/GRU collapsed into pt_recurrent.
    expected = [
        "pt_tensor_create", "pt_tensor_op", "pt_tensor_reshape",
        "pt_tensor_info", "pt_print_tensor",
        "pt_recurrent",
        "pt_pretrained_backbone",
        "pt_freeze_backbone", "pt_model_info",
    ]
    for tn in expected:
        assert tn in NODE_REGISTRY, f"Missing: {tn}"

def test_subcategories_set():
    from nodes import NODE_REGISTRY
    checks = {
        "pt_tensor_op":           "",
        "pt_tensor_reshape":      "",
        "pt_recurrent":           "Recurrent",
        "pt_pretrained_backbone": "Pretrained",
        "pt_freeze_backbone":     "Pretrained",
    }
    for tn, expected_sub in checks.items():
        cls = NODE_REGISTRY[tn]
        assert cls.subcategory == expected_sub, f"{tn}: expected '{expected_sub}', got '{cls.subcategory}'"
