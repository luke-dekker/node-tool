"""PyTorch optimizer/loss/scheduler factories — plugin-owned, GUI-agnostic.

These used to live in gui/mixins/training.py. They belong to the pytorch plugin
because they import torch. Any frontend can import them by string key; the
frontend never needs to know torch exists.
"""
from __future__ import annotations
from typing import Any


def build_optimizer(name: str, model: Any, lr: float,
                    weight_decay: float = 0.0, momentum: float = 0.9):
    """Construct a torch optimizer from a string name."""
    import torch.optim as optim
    if model is None:
        return None
    params = model.parameters()
    key = name.strip().lower().replace("_", "").replace(" ", "")
    if key == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    if key == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)  # default


def build_loss(name: str):
    """Construct a torch loss function from a string name."""
    import torch.nn as nn
    key = name.strip().lower().replace("_", "").replace(" ", "").replace("-", "")
    return {
        "mse":           nn.MSELoss(),
        "bce":           nn.BCELoss(),
        "bcewithlogits": nn.BCEWithLogitsLoss(),
        "l1":            nn.L1Loss(),
    }.get(key, nn.CrossEntropyLoss())  # default


OPTIMIZER_CHOICES = ["adam", "adamw", "sgd", "rmsprop"]
LOSS_CHOICES      = ["crossentropy", "mse", "bce", "bce_with_logits", "l1"]
