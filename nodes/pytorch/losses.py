"""PyTorch loss function nodes."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Training"


class MSELossNode(BaseNode):
    type_name = "pt_mse_loss"
    label = "MSE Loss"
    category = CATEGORY
    description = "nn.MSELoss(reduction)"

    def _setup_ports(self):
        self.add_input("reduction", PortType.STRING, default="mean")
        self.add_output("loss_fn", PortType.LOSS_FN)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            return {"loss_fn": nn.MSELoss(reduction=str(inputs.get("reduction", "mean")))}
        except Exception:
            return {"loss_fn": None}


class CrossEntropyLossNode(BaseNode):
    type_name = "pt_cross_entropy"
    label = "CrossEntropy Loss"
    category = CATEGORY
    description = "nn.CrossEntropyLoss()"

    def _setup_ports(self):
        self.add_output("loss_fn", PortType.LOSS_FN)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            return {"loss_fn": nn.CrossEntropyLoss()}
        except Exception:
            return {"loss_fn": None}


class BCELossNode(BaseNode):
    type_name = "pt_bce_loss"
    label = "BCE Loss"
    category = CATEGORY
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


class BCEWithLogitsNode(BaseNode):
    type_name = "pt_bce_logits"
    label = "BCEWithLogits Loss"
    category = CATEGORY
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


class L1LossNode(BaseNode):
    type_name = "pt_l1_loss"
    label = "L1 Loss"
    category = CATEGORY
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


# Subcategory stamp
_SC = "Losses"
from core.node import BaseNode as _BN
for _n, _c in list(globals().items()):
    if isinstance(_c, type) and issubclass(_c, _BN) and _c is not _BN:
        _c.subcategory = _SC
