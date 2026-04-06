"""L1 Loss node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class L1LossNode(BaseNode):
    type_name   = "pt_l1_loss"
    label       = "L1 Loss"
    category    = "Training"
    subcategory = "Losses"
    description = "nn.L1Loss(reduction)"

    def _setup_ports(self):
        self.add_input("reduction", PortType.STRING, default="mean")
        self.add_output("loss_fn", PortType.LOSS_FN)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            return {"loss_fn": nn.L1Loss(reduction=str(inputs.get("reduction", "mean")))}
        except Exception:
            return {"loss_fn": None}

    def export(self, iv, ov):
        return ["import torch.nn as nn"], [f"{ov['loss_fn']} = nn.L1Loss(reduction={self._val(iv,'reduction')})"]
