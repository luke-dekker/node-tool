"""PyTorch LR scheduler nodes."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Training"


class StepLRNode(BaseNode):
    type_name = "pt_step_lr"
    label = "StepLR"
    category = CATEGORY
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


class MultiStepLRNode(BaseNode):
    type_name = "pt_multistep_lr"
    label = "MultiStepLR"
    category = CATEGORY
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


class ExponentialLRNode(BaseNode):
    type_name = "pt_exponential_lr"
    label = "ExponentialLR"
    category = CATEGORY
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


class CosineAnnealingLRNode(BaseNode):
    type_name = "pt_cosine_lr"
    label = "CosineAnnealingLR"
    category = CATEGORY
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


class ReduceLROnPlateauNode(BaseNode):
    type_name = "pt_reduce_lr_plateau"
    label = "ReduceLROnPlateau"
    category = CATEGORY
    description = "Reduce LR when a metric stops improving. Requires val_dataloader."

    def _setup_ports(self):
        self.add_input("optimizer", PortType.OPTIMIZER, default=None)
        self.add_input("mode",      PortType.STRING,    default="min")
        self.add_input("factor",    PortType.FLOAT,     default=0.1)
        self.add_input("patience",  PortType.INT,       default=10)
        self.add_output("scheduler", PortType.SCHEDULER)

    def execute(self, inputs):
        try:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            opt = inputs.get("optimizer")
            if opt is None:
                return {"scheduler": None}
            return {"scheduler": ReduceLROnPlateau(
                opt,
                mode=str(inputs.get("mode", "min") or "min"),
                factor=float(inputs.get("factor", 0.1)),
                patience=int(inputs.get("patience", 10)),
            )}
        except Exception:
            return {"scheduler": None}


# Subcategory stamp
_SC = "Schedulers"
from core.node import BaseNode as _BN
for _n, _c in list(globals().items()):
    if isinstance(_c, type) and issubclass(_c, _BN) and _c is not _BN:
        _c.subcategory = _SC
