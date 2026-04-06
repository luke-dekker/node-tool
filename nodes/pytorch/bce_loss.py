"""BCE Loss node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class BCELossNode(BaseNode):
    type_name   = "pt_bce_loss"
    label       = "BCE Loss"
    category    = "Training"
    subcategory = "Losses"
    description = "nn.BCELoss(reduction)"

    def _setup_ports(self):
        self.add_input("reduction", PortType.STRING, default="mean")
        self.add_output("loss_fn", PortType.LOSS_FN)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            return {"loss_fn": nn.BCELoss(reduction=str(inputs.get("reduction", "mean")))}
        except Exception:
            return {"loss_fn": None}

    def export(self, iv, ov):
        return ["import torch.nn as nn"], [f"{ov['loss_fn']} = nn.BCELoss(reduction={self._val(iv,'reduction')})"]
