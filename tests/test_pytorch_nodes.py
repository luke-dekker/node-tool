"""Tests for pytorch nodes - no DPG, no display required."""
import pytest


def test_linear_node():
    from nodes.pytorch import LinearNode
    import torch, torch.nn as nn
    node = LinearNode()
    node.execute({"tensor_in": None, "in_features": 4, "out_features": 2,
                  "bias": True, "activation": "none", "freeze": False})
    assert isinstance(node._layer, nn.Linear)
    assert node._layer.in_features == 4


def test_linear_node_tensor_out():
    from nodes.pytorch import LinearNode
    import torch
    node = LinearNode()
    result = node.execute({"tensor_in": torch.randn(1, 4), "in_features": 4,
                           "out_features": 2, "bias": True, "activation": "none", "freeze": False})
    assert result["tensor_out"].shape == (1, 2)


def test_rand_tensor_node():
    from nodes.pytorch import RandTensorNode
    import torch
    result = RandTensorNode().execute({"shape": "2,3", "requires_grad": False})
    assert isinstance(result["tensor"], torch.Tensor)
    assert list(result["tensor"].shape) == [2, 3]


def test_mse_loss():
    from nodes.pytorch import MSELossNode
    import torch.nn as nn
    result = MSELossNode().execute({"reduction": "mean"})
    assert isinstance(result["loss_fn"], nn.MSELoss)


def test_adam_none_guard():
    from nodes.pytorch import AdamNode
    result = AdamNode().execute({"model": None, "lr": 0.001, "weight_decay": 0.0})
    assert result["optimizer"] is None


def test_step_lr_node():
    import torch.nn as nn, torch.optim as optim
    from nodes.pytorch import AdamNode, StepLRNode
    model = nn.Linear(4, 2)
    opt = AdamNode().execute({"model": model, "lr": 0.01, "weight_decay": 0.0})["optimizer"]
    result = StepLRNode().execute({"optimizer": opt, "step_size": 5, "gamma": 0.5})
    from torch.optim.lr_scheduler import StepLR
    assert isinstance(result["scheduler"], StepLR)


def test_step_lr_none_guard():
    from nodes.pytorch import StepLRNode
    result = StepLRNode().execute({"optimizer": None, "step_size": 5, "gamma": 0.1})
    assert result["scheduler"] is None


def test_cosine_annealing_node():
    import torch.nn as nn, torch.optim as optim
    from nodes.pytorch import AdamNode, CosineAnnealingLRNode
    model = nn.Linear(2, 1)
    opt = AdamNode().execute({"model": model, "lr": 0.01, "weight_decay": 0.0})["optimizer"]
    result = CosineAnnealingLRNode().execute({"optimizer": opt, "T_max": 10, "eta_min": 1e-6})
    from torch.optim.lr_scheduler import CosineAnnealingLR
    assert isinstance(result["scheduler"], CosineAnnealingLR)


def test_reduce_lr_plateau_node():
    import torch.nn as nn, torch.optim as optim
    from nodes.pytorch import AdamNode, ReduceLROnPlateauNode
    model = nn.Linear(2, 1)
    opt = AdamNode().execute({"model": model, "lr": 0.01, "weight_decay": 0.0})["optimizer"]
    result = ReduceLROnPlateauNode().execute({"optimizer": opt, "mode": "min",
                                              "factor": 0.1, "patience": 5})
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    assert isinstance(result["scheduler"], ReduceLROnPlateau)


def test_tensor_info():
    from nodes.pytorch import TensorInfoNode, RandTensorNode
    tensor = RandTensorNode().execute({"shape": "3,4", "requires_grad": False})["tensor"]
    info = TensorInfoNode().execute({"tensor": tensor})["info"]
    assert "3" in info and "4" in info
