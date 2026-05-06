"""LR Scheduler factory node — replaces StepLRNode, MultiStepLRNode, ExponentialLRNode,
CosineAnnealingLRNode, and ReduceLROnPlateauNode with a single dropdown-driven node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class LRSchedulerNode(BaseNode):
    type_name   = "pt_lr_scheduler"
    label       = "LR Scheduler"
    category    = "Training"
    subcategory = "Schedulers"
    description = "Construct a PyTorch LR scheduler selected by dropdown."

    def _setup_ports(self):
        self.add_input("optimizer",      PortType.OPTIMIZER, default=None)
        self.add_input("scheduler_type", PortType.STRING,    default="step",
                       choices=["step", "multistep", "exponential", "cosine", "plateau"])
        # StepLR
        self.add_input("step_size",  PortType.INT,   default=10)
        # MultiStepLR
        self.add_input("milestones", PortType.STRING, default="30,60,90")
        # shared: StepLR / MultiStepLR / ExponentialLR
        self.add_input("gamma",      PortType.FLOAT,  default=0.1)
        # CosineAnnealingLR
        self.add_input("T_max",      PortType.INT,    default=10)
        self.add_input("eta_min",    PortType.FLOAT,  default=0.0)
        # ReduceLROnPlateau
        self.add_input("mode",       PortType.STRING, default="min")
        self.add_input("factor",     PortType.FLOAT,  default=0.1)
        self.add_input("patience",   PortType.INT,    default=10)
        self.add_output("scheduler", PortType.SCHEDULER)

    def execute(self, inputs):
        try:
            import torch.optim.lr_scheduler as sched_mod
            opt = inputs.get("optimizer")
            if opt is None:
                return {"scheduler": None}
            stype = str(inputs.get("scheduler_type", "step") or "step")

            if stype == "step":
                return {"scheduler": sched_mod.StepLR(
                    opt,
                    step_size=int(inputs.get("step_size", 10)),
                    gamma=float(inputs.get("gamma", 0.1)),
                )}
            elif stype == "multistep":
                ms_str = inputs.get("milestones", "30,60,90") or "30,60,90"
                milestones = [int(x) for x in ms_str.split(",") if x.strip()]
                return {"scheduler": sched_mod.MultiStepLR(
                    opt,
                    milestones=milestones,
                    gamma=float(inputs.get("gamma", 0.1)),
                )}
            elif stype == "exponential":
                return {"scheduler": sched_mod.ExponentialLR(
                    opt,
                    gamma=float(inputs.get("gamma", 0.1)),
                )}
            elif stype == "cosine":
                return {"scheduler": sched_mod.CosineAnnealingLR(
                    opt,
                    T_max=int(inputs.get("T_max", 10)),
                    eta_min=float(inputs.get("eta_min", 0.0)),
                )}
            elif stype == "plateau":
                return {"scheduler": sched_mod.ReduceLROnPlateau(
                    opt,
                    mode=str(inputs.get("mode", "min") or "min"),
                    factor=float(inputs.get("factor", 0.1)),
                    patience=int(inputs.get("patience", 10)),
                )}
            return {"scheduler": None}
        except Exception:
            return {"scheduler": None}

    def export(self, iv, ov):
        opt = self._val(iv, 'optimizer')
        stype = (self.inputs["scheduler_type"].default_value or "step")

        if stype == "step":
            ss = self._val(iv, 'step_size')
            g = self._val(iv, 'gamma')
            return ["from torch.optim.lr_scheduler import StepLR"], [
                f"{ov['scheduler']} = StepLR({opt}, step_size={ss}, gamma={g})"
            ]
        elif stype == "multistep":
            ms = self._val(iv, 'milestones')
            g = self._val(iv, 'gamma')
            return ["from torch.optim.lr_scheduler import MultiStepLR"], [
                f"{ov['scheduler']} = MultiStepLR({opt}, milestones={ms}, gamma={g})"
            ]
        elif stype == "exponential":
            g = self._val(iv, 'gamma')
            return ["from torch.optim.lr_scheduler import ExponentialLR"], [
                f"{ov['scheduler']} = ExponentialLR({opt}, gamma={g})"
            ]
        elif stype == "cosine":
            t = self._val(iv, 'T_max')
            eta = self._val(iv, 'eta_min')
            return ["from torch.optim.lr_scheduler import CosineAnnealingLR"], [
                f"{ov['scheduler']} = CosineAnnealingLR({opt}, T_max={t}, eta_min={eta})"
            ]
        elif stype == "plateau":
            mode = self._val(iv, 'mode')
            factor = self._val(iv, 'factor')
            patience = self._val(iv, 'patience')
            return ["from torch.optim.lr_scheduler import ReduceLROnPlateau"], [
                f"{ov['scheduler']} = ReduceLROnPlateau({opt}, mode={mode}, factor={factor}, patience={patience})"
            ]
        return [], []
