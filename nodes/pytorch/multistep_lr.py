"""MultiStepLR scheduler node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class MultiStepLRNode(BaseNode):
    type_name   = "pt_multistep_lr"
    label       = "MultiStepLR"
    category    = "Training"
    subcategory = "Schedulers"
    description = "Decay LR at milestones (comma-separated epoch numbers)."

    def _setup_ports(self):
        self.add_input("optimizer",  PortType.OPTIMIZER, default=None)
        self.add_input("milestones", PortType.STRING,    default="30,60,90")
        self.add_input("gamma",      PortType.FLOAT,     default=0.1)
        self.add_output("scheduler", PortType.SCHEDULER)

    def execute(self, inputs):
        try:
            from torch.optim.lr_scheduler import MultiStepLR
            opt = inputs.get("optimizer")
            if opt is None:
                return {"scheduler": None}
            ms_str = inputs.get("milestones", "30,60,90") or "30,60,90"
            milestones = [int(x) for x in ms_str.split(",") if x.strip()]
            return {"scheduler": MultiStepLR(opt, milestones=milestones,
                                             gamma=float(inputs["gamma"]))}
        except Exception:
            return {"scheduler": None}

    def export(self, iv, ov):
        opt = self._val(iv, 'optimizer')
        ms = self._val(iv, 'milestones')
        g = self._val(iv, 'gamma')
        return ["from torch.optim.lr_scheduler import MultiStepLR"], [
            f"{ov['scheduler']} = MultiStepLR({opt}, milestones={ms}, gamma={g})"
        ]
