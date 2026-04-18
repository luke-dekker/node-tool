"""Port types registered by the PyTorch plugin.

These used to live in core/port_types.py (under a "transition period" comment).
Now they live with the plugin that owns them: plugins/pytorch/__init__.py
calls register_all() at load time, before its nodes are discovered.

Core stays pure — no torch, no domain-specific registration at import time.
"""
from __future__ import annotations
from typing import Any

from core.port_types import PortTypeRegistry


def _coerce_tensor(value: Any) -> Any:
    """Coerce scalars/lists to torch.Tensor."""
    if isinstance(value, (int, float, list)):
        try:
            import torch
            return torch.tensor(value)
        except Exception:
            return None
    return value


def register_all() -> None:
    """Register every PyTorch port type. Called from plugins/pytorch/__init__.py."""
    PortTypeRegistry.register("TENSOR",     default=None, coerce=_coerce_tensor,
                              color=(255, 120, 40, 255),  pin_shape="circle",
                              description="torch.Tensor")
    PortTypeRegistry.register("MODULE",     default=None,
                              color=(160, 80, 255, 255),  pin_shape="triangle",
                              description="torch.nn.Module")
    PortTypeRegistry.register("DATALOADER", default=None,
                              color=(40, 200, 200, 255),  pin_shape="quad",
                              description="torch.utils.data.DataLoader")
    PortTypeRegistry.register("OPTIMIZER",  default=None,
                              color=(255, 200, 40, 255),  pin_shape="circle",
                              description="torch.optim.Optimizer")
    PortTypeRegistry.register("LOSS_FN",    default=None,
                              color=(220, 60, 120, 255),  pin_shape="circle",
                              description="Loss function callable")
    PortTypeRegistry.register("SCHEDULER",  default=None,
                              color=(160, 230, 60, 255),  pin_shape="circle_filled",
                              description="Learning rate scheduler")
    PortTypeRegistry.register("DATASET",    default=None,
                              color=(80, 220, 180, 255),  pin_shape="circle_filled",
                              description="torch.utils.data.Dataset")
    PortTypeRegistry.register("TRANSFORM",  default=None,
                              color=(200, 130, 255, 255), pin_shape="circle_filled",
                              description="Data transform callable")
