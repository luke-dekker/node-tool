"""StepLR scheduler node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class StepLRNode(BaseNode):
    type_name   = "pt_step_lr"
    label       = "StepLR"
    category    = "Training"
    subcategory = "Schedulers"
    description = "Decay LR by gamma every step_size epochs."

    def _setup_ports(self):
        self.add_input("optimizer", PortType.OPTIMIZER, default=None)
        self.add_input("step_size", PortType.INT,   default=10)
        self.add_input("gamma",     PortType.FLOAT, default=0.1)
        self.add_output("scheduler", PortType.SCHEDULER)

    def execute(self, inputs):
        try:
            from torch.optim.lr_scheduler import StepLR
            opt = inputs.get("optimizer")
            if opt is None:
                return {"scheduler": None}
            return {"scheduler": StepLR(opt, step_size=int(inputs["step_size"]),
                                        gamma=float(inputs["gamma"]))}
        except Exception:
            return {"scheduler": None}

    def export(self, iv, ov):
        opt = self._val(iv, 'optimizer')
        ss = self._val(iv, 'step_size')
        g = self._val(iv, 'gamma')
        return ["from torch.optim.lr_scheduler import StepLR"], [
            f"{ov['scheduler']} = StepLR({opt}, step_size={ss}, gamma={g})"
        ]
