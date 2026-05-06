"""Loss Function factory node — replaces MSELossNode, CrossEntropyLossNode, BCELossNode,
BCEWithLogitsNode, and L1LossNode with a single dropdown-driven node."""
from __future__ import annotations
from core.node import BaseNode, PortType

_LOSS_MAP = {
    "mse":           ("MSELoss",            lambda nn, r: nn.MSELoss(reduction=r)),
    "cross_entropy": ("CrossEntropyLoss",   lambda nn, r: nn.CrossEntropyLoss()),
    "bce":           ("BCELoss",            lambda nn, r: nn.BCELoss(reduction=r)),
    "bce_logits":    ("BCEWithLogitsLoss",  lambda nn, r: nn.BCEWithLogitsLoss(reduction=r)),
    "l1":            ("L1Loss",             lambda nn, r: nn.L1Loss(reduction=r)),
}


class LossFnNode(BaseNode):
    type_name   = "pt_loss_fn"
    label       = "Loss Function"
    category    = "Training"
    subcategory = "Loss"
    description = "Construct a PyTorch loss function selected by dropdown."

    def _setup_ports(self):
        self.add_input("loss_type",  PortType.STRING, default="mse",
                       choices=["mse", "cross_entropy", "bce", "bce_logits", "l1"])
        self.add_input("reduction",  PortType.STRING, default="mean",
                       choices=["mean", "sum", "none"])
        self.add_output("loss_fn", PortType.LOSS_FN)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            loss_type = str(inputs.get("loss_type", "mse") or "mse")
            reduction = str(inputs.get("reduction", "mean") or "mean")
            entry = _LOSS_MAP.get(loss_type)
            if entry is None:
                return {"loss_fn": None}
            _, factory = entry
            return {"loss_fn": factory(nn, reduction)}
        except Exception:
            return {"loss_fn": None}

    def export(self, iv, ov):
        loss_type = (self.inputs["loss_type"].default_value or "mse")
        reduction = (self.inputs["reduction"].default_value or "mean")
        entry = _LOSS_MAP.get(loss_type)
        cls_name = entry[0] if entry else "MSELoss"
        no_reduction = loss_type == "cross_entropy"
        args = "" if no_reduction else f"reduction={repr(reduction)}"
        return ["import torch.nn as nn"], [
            f"{ov['loss_fn']} = nn.{cls_name}({args})"
        ]
