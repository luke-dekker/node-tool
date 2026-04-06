"""SGD optimizer node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class SGDNode(BaseNode):
    type_name   = "pt_sgd"
    label       = "SGD"
    category    = "Training"
    subcategory = "Optimizers"
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

    def export(self, iv, ov):
        m = self._val(iv, 'model')
        lr = self._val(iv, 'lr')
        mom = self._val(iv, 'momentum')
        wd = self._val(iv, 'weight_decay')
        return ["import torch.optim as optim"], [
            f"{ov['optimizer']} = optim.SGD({m}.parameters(), lr={lr}, momentum={mom}, weight_decay={wd})"
        ]
