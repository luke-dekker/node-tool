"""Adam optimizer node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class AdamNode(BaseNode):
    type_name   = "pt_adam"
    label       = "Adam"
    category    = "Training"
    subcategory = "Optimizers"
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

    def export(self, iv, ov):
        m = self._val(iv, 'model')
        lr = self._val(iv, 'lr')
        wd = self._val(iv, 'weight_decay')
        return ["import torch.optim as optim"], [
            f"{ov['optimizer']} = optim.Adam({m}.parameters(), lr={lr}, weight_decay={wd})"
        ]
