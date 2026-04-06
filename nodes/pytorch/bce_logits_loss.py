"""BCEWithLogits Loss node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class BCEWithLogitsNode(BaseNode):
    type_name   = "pt_bce_logits"
    label       = "BCEWithLogits Loss"
    category    = "Training"
    subcategory = "Losses"
    description = "nn.BCEWithLogitsLoss(reduction)"

    def _setup_ports(self):
        self.add_input("reduction", PortType.STRING, default="mean")
        self.add_output("loss_fn", PortType.LOSS_FN)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            return {"loss_fn": nn.BCEWithLogitsLoss(reduction=str(inputs.get("reduction", "mean")))}
        except Exception:
            return {"loss_fn": None}

    def export(self, iv, ov):
        return ["import torch.nn as nn"], [f"{ov['loss_fn']} = nn.BCEWithLogitsLoss(reduction={self._val(iv,'reduction')})"]
