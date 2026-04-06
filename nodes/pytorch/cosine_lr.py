"""CosineAnnealingLR scheduler node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class CosineAnnealingLRNode(BaseNode):
    type_name   = "pt_cosine_lr"
    label       = "CosineAnnealingLR"
    category    = "Training"
    subcategory = "Schedulers"
    description = "Cosine annealing from base LR to eta_min over T_max epochs."

    def _setup_ports(self):
        self.add_input("optimizer", PortType.OPTIMIZER, default=None)
        self.add_input("T_max",     PortType.INT,       default=10)
        self.add_input("eta_min",   PortType.FLOAT,     default=0.0)
        self.add_output("scheduler", PortType.SCHEDULER)

    def execute(self, inputs):
        try:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            opt = inputs.get("optimizer")
            if opt is None:
                return {"scheduler": None}
            return {"scheduler": CosineAnnealingLR(opt,
                                                    T_max=int(inputs["T_max"]),
                                                    eta_min=float(inputs["eta_min"]))}
        except Exception:
            return {"scheduler": None}

    def export(self, iv, ov):
        opt = self._val(iv, 'optimizer')
        t = self._val(iv, 'T_max')
        eta = self._val(iv, 'eta_min')
        return ["from torch.optim.lr_scheduler import CosineAnnealingLR"], [
            f"{ov['scheduler']} = CosineAnnealingLR({opt}, T_max={t}, eta_min={eta})"
        ]
