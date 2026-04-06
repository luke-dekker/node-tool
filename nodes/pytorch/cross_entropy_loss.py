"""CrossEntropy Loss node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class CrossEntropyLossNode(BaseNode):
    type_name   = "pt_cross_entropy"
    label       = "CrossEntropy Loss"
    category    = "Training"
    subcategory = "Losses"
    description = "nn.CrossEntropyLoss()"

    def _setup_ports(self):
        self.add_output("loss_fn", PortType.LOSS_FN)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            return {"loss_fn": nn.CrossEntropyLoss()}
        except Exception:
            return {"loss_fn": None}

    def export(self, iv, ov):
        return ["import torch.nn as nn"], [f"{ov['loss_fn']} = nn.CrossEntropyLoss()"]
