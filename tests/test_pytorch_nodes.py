"""Tests for pytorch nodes - no DPG, no display required."""
import pytest


def test_linear_node():
    # LinearNode is now an alias for LayerNode; in_features is shape-inferred
    # from the upstream tensor. Provide one to test materialization.
    from nodes.pytorch import LinearNode
    import torch, torch.nn as nn
    node = LinearNode()
    node.execute({"tensor_in": torch.randn(1, 4), "kind": "linear",
                  "out_features": 2, "bias": True, "activation": "none", "freeze": False})
    assert isinstance(node._layer, nn.Linear)
    assert node._layer.in_features == 4


def test_linear_node_tensor_out():
    from nodes.pytorch import LinearNode
    import torch
    node = LinearNode()
    result = node.execute({"tensor_in": torch.randn(1, 4), "kind": "linear",
                           "out_features": 2, "bias": True, "activation": "none", "freeze": False})
    assert result["tensor_out"].shape == (1, 2)


def test_rand_tensor_node():
    from nodes.pytorch.tensor_create import TensorCreateNode
    import torch
    result = TensorCreateNode().execute({"fill": "rand", "shape": "2,3", "requires_grad": False})
    assert isinstance(result["tensor"], torch.Tensor)
    assert list(result["tensor"].shape) == [2, 3]


def test_loss_fn_node():
    from nodes.pytorch import LossFnNode
    import torch.nn as nn
    result = LossFnNode().execute({"loss_type": "mse", "reduction": "mean"})
    assert isinstance(result["loss_fn"], nn.MSELoss)
    result = LossFnNode().execute({"loss_type": "cross_entropy", "reduction": "mean"})
    assert isinstance(result["loss_fn"], nn.CrossEntropyLoss)


def test_optimizer_none_guard():
    from nodes.pytorch import OptimizerNode
    result = OptimizerNode().execute({"model": None, "optimizer_type": "adam",
                                      "lr": 0.001, "weight_decay": 0.0, "momentum": 0.9})
    assert result["optimizer"] is None


def test_optimizer_node():
    import torch.nn as nn
    from nodes.pytorch import OptimizerNode
    model = nn.Linear(4, 2)
    result = OptimizerNode().execute({"model": model, "optimizer_type": "adam",
                                      "lr": 0.01, "weight_decay": 0.0, "momentum": 0.9})
    assert result["optimizer"] is not None


def test_lr_scheduler_node():
    import torch.nn as nn
    from nodes.pytorch import OptimizerNode, LRSchedulerNode
    model = nn.Linear(4, 2)
    opt = OptimizerNode().execute({"model": model, "optimizer_type": "adam",
                                   "lr": 0.01, "weight_decay": 0.0, "momentum": 0.9})["optimizer"]
    result = LRSchedulerNode().execute({"optimizer": opt, "scheduler_type": "step",
                                        "step_size": 5, "gamma": 0.5,
                                        "milestones": "30,60", "T_max": 10,
                                        "eta_min": 0.0, "mode": "min",
                                        "factor": 0.1, "patience": 10})
    from torch.optim.lr_scheduler import StepLR
    assert isinstance(result["scheduler"], StepLR)


def test_lr_scheduler_none_guard():
    from nodes.pytorch import LRSchedulerNode
    result = LRSchedulerNode().execute({"optimizer": None, "scheduler_type": "step",
                                        "step_size": 5, "gamma": 0.1,
                                        "milestones": "", "T_max": 10,
                                        "eta_min": 0.0, "mode": "min",
                                        "factor": 0.1, "patience": 10})
    assert result["scheduler"] is None


def test_tensor_info():
    from nodes.pytorch import TensorInfoNode
    from nodes.pytorch.tensor_create import TensorCreateNode
    tensor = TensorCreateNode().execute({"fill": "rand", "shape": "3,4", "requires_grad": False})["tensor"]
    info = TensorInfoNode().execute({"tensor": tensor})["info"]
    assert "3" in info and "4" in info
