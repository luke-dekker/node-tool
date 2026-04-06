"""ExponentialLR scheduler node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ExponentialLRNode(BaseNode):
    type_name   = "pt_exponential_lr"
    label       = "ExponentialLR"
    category    = "Training"
    subcategory = "Schedulers"
    description = "Multiply LR by gamma every epoch."

    def _setup_ports(self):
        self.add_input("optimizer", PortType.OPTIMIZER, default=None)
        self.add_input("gamma",     PortType.FLOAT,     default=0.9)
        self.add_output("scheduler", PortType.SCHEDULER)

    def execute(self, inputs):
        try:
            from torch.optim.lr_scheduler import ExponentialLR
            opt = inputs.get("optimizer")
            if opt is None:
                return {"scheduler": None}
            return {"scheduler": ExponentialLR(opt, gamma=float(inputs["gamma"]))}
        except Exception:
            return {"scheduler": None}

    def export(self, iv, ov):
        opt = self._val(iv, 'optimizer')
        g = self._val(iv, 'gamma')
        return ["from torch.optim.lr_scheduler import ExponentialLR"], [
            f"{ov['scheduler']} = ExponentialLR({opt}, gamma={g})"
        ]
