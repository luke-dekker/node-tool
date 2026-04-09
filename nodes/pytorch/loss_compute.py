"""LossComputeNode — compute a loss tensor inline as a graph operation.

The bridge that lets you compute losses INSIDE the graph instead of relying
on TrainingConfigNode's string-based loss menu. Useful for:

  - VAE-style training where the loss is `recon_loss + beta * KL` and needs
    multiple model outputs to assemble
  - Multi-task learning where each task computes its own loss tensor and a
    Math node sums them with optional weights
  - Custom losses with extra arguments
  - Anything where you want to inspect the loss as a real graph value

Used together with `TrainingConfigNode(loss_is_output=True)`: the graph's
leaf is the scalar loss tensor; the training loop just calls
`loss = model(batch); loss.backward()` with no separate loss_fn.

Supported loss types:
    mse           — F.mse_loss(pred, target)
    bce           — F.binary_cross_entropy(pred, target) — pred must be in [0,1]
    bce_logits    — F.binary_cross_entropy_with_logits(pred, target)
    cross_entropy — F.cross_entropy(pred, target) — pred is logits, target is class indices
    l1            — F.l1_loss(pred, target)
    smooth_l1     — F.smooth_l1_loss(pred, target) (Huber)
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


_LOSS_TYPES = ["mse", "bce", "bce_logits", "cross_entropy", "l1", "smooth_l1"]


def _apply_loss(loss_type: str, pred, target):
    """Dispatch to the right F.* loss function."""
    import torch.nn.functional as F
    key = (loss_type or "mse").strip().lower().replace("-", "_")
    if key == "mse":
        return F.mse_loss(pred, target)
    if key == "bce":
        return F.binary_cross_entropy(pred, target)
    if key in ("bce_logits", "bcewithlogits"):
        return F.binary_cross_entropy_with_logits(pred, target)
    if key in ("cross_entropy", "crossentropy", "ce"):
        return F.cross_entropy(pred, target)
    if key == "l1":
        return F.l1_loss(pred, target)
    if key in ("smooth_l1", "huber"):
        return F.smooth_l1_loss(pred, target)
    return F.mse_loss(pred, target)


class LossComputeNode(BaseNode):
    type_name   = "pt_loss_compute"
    label       = "Loss Compute"
    category    = "Training"
    subcategory = "Loss"
    description = (
        "Compute a loss tensor as a first-class graph operation. Wire pred + "
        "target tensors and pick a loss type. Output is a scalar loss tensor "
        "that can be summed with other losses (multi-task) or fed directly "
        "into TrainingConfig with loss_is_output=True."
    )

    def _setup_ports(self) -> None:
        self.add_input("pred",      PortType.TENSOR, default=None,
                       description="Model prediction tensor")
        self.add_input("target",    PortType.TENSOR, default=None,
                       description="Ground truth tensor (or input for autoencoder reconstruction)")
        self.add_input("loss_type", PortType.STRING, default="mse",
                       choices=_LOSS_TYPES)
        self.add_input("weight",    PortType.FLOAT,  default=1.0,
                       description="Multiplier on the loss before output — useful for multi-task weighting")
        self.add_output("loss", PortType.TENSOR,
                        description="Scalar loss tensor")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pred   = inputs.get("pred")
        target = inputs.get("target")
        loss_type = (inputs.get("loss_type") or "mse").strip().lower()
        weight = float(inputs.get("weight") or 1.0)
        if pred is None or target is None:
            return {"loss": None}
        try:
            loss = _apply_loss(loss_type, pred, target)
            if weight != 1.0:
                loss = loss * weight
            return {"loss": loss}
        except Exception:
            return {"loss": None}

    def export(self, iv, ov):
        pred   = iv.get("pred")   or "None  # TODO: connect pred"
        target = iv.get("target") or "None  # TODO: connect target"
        loss_type = (self.inputs["loss_type"].default_value or "mse").strip().lower().replace("-", "_")
        weight = float(self.inputs["weight"].default_value or 1.0)
        out = ov.get("loss", "_loss")

        loss_call_map = {
            "mse":           f"F.mse_loss({pred}, {target})",
            "bce":           f"F.binary_cross_entropy({pred}, {target})",
            "bce_logits":    f"F.binary_cross_entropy_with_logits({pred}, {target})",
            "cross_entropy": f"F.cross_entropy({pred}, {target})",
            "l1":            f"F.l1_loss({pred}, {target})",
            "smooth_l1":     f"F.smooth_l1_loss({pred}, {target})",
        }
        call = loss_call_map.get(loss_type, loss_call_map["mse"])
        if weight != 1.0:
            return ["import torch", "import torch.nn.functional as F"], [
                f"{out} = {call} * {weight}",
            ]
        return ["import torch", "import torch.nn.functional as F"], [
            f"{out} = {call}",
        ]
