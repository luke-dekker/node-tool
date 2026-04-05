"""PyTorch optimizer nodes."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Training"


class AdamNode(BaseNode):
    type_name = "pt_adam"
    label = "Adam"
    category = CATEGORY
    description = "torch.optim.Adam(model.parameters(), lr, weight_decay)"

    def _setup_ports(self):
        self.add_input("model", PortType.MODULE, default=None)
        self.add_input("lr", PortType.FLOAT, default=0.001)
        self.add_input("weight_decay", PortType.FLOAT, default=0.0)
        self.add_output("optimizer", PortType.OPTIMIZER)

    def execute(self, inputs):
        try:
            model = inputs.get("model")
            if model is None:
                return {"optimizer": None}
            import torch.optim as optim
            return {"optimizer": optim.Adam(
                model.parameters(),
                lr=float(inputs.get("lr", 0.001)),
                weight_decay=float(inputs.get("weight_decay", 0.0))
            )}
        except Exception:
            return {"optimizer": None}


class SGDNode(BaseNode):
    type_name = "pt_sgd"
    label = "SGD"
    category = CATEGORY
    description = "torch.optim.SGD(model.parameters(), lr, momentum, weight_decay)"

    def _setup_ports(self):
        self.add_input("model", PortType.MODULE, default=None)
        self.add_input("lr", PortType.FLOAT, default=0.01)
        self.add_input("momentum", PortType.FLOAT, default=0.9)
        self.add_input("weight_decay", PortType.FLOAT, default=0.0)
        self.add_output("optimizer", PortType.OPTIMIZER)

    def execute(self, inputs):
        try:
            model = inputs.get("model")
            if model is None:
                return {"optimizer": None}
            import torch.optim as optim
            return {"optimizer": optim.SGD(
                model.parameters(),
                lr=float(inputs.get("lr", 0.01)),
                momentum=float(inputs.get("momentum", 0.9)),
                weight_decay=float(inputs.get("weight_decay", 0.0))
            )}
        except Exception:
            return {"optimizer": None}


class AdamWNode(BaseNode):
    type_name = "pt_adamw"
    label = "AdamW"
    category = CATEGORY
    description = "torch.optim.AdamW(model.parameters(), lr, weight_decay)"

    def _setup_ports(self):
        self.add_input("model", PortType.MODULE, default=None)
        self.add_input("lr", PortType.FLOAT, default=0.001)
        self.add_input("weight_decay", PortType.FLOAT, default=0.01)
        self.add_output("optimizer", PortType.OPTIMIZER)

    def execute(self, inputs):
        try:
            model = inputs.get("model")
            if model is None:
                return {"optimizer": None}
            import torch.optim as optim
            return {"optimizer": optim.AdamW(
                model.parameters(),
                lr=float(inputs.get("lr", 0.001)),
                weight_decay=float(inputs.get("weight_decay", 0.01))
            )}
        except Exception:
            return {"optimizer": None}


# Subcategory stamp
_SC = "Optimizers"
from core.node import BaseNode as _BN
for _n, _c in list(globals().items()):
    if isinstance(_c, type) and issubclass(_c, _BN) and _c is not _BN:
        _c.subcategory = _SC
