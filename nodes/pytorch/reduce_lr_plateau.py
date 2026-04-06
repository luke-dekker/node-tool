"""ReduceLROnPlateau scheduler node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ReduceLROnPlateauNode(BaseNode):
    type_name   = "pt_reduce_lr_plateau"
    label       = "ReduceLROnPlateau"
    category    = "Training"
    subcategory = "Schedulers"
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

    def export(self, iv, ov):
        opt = self._val(iv, 'optimizer')
        mode = self._val(iv, 'mode')
        factor = self._val(iv, 'factor')
        patience = self._val(iv, 'patience')
        return ["from torch.optim.lr_scheduler import ReduceLROnPlateau"], [
            f"{ov['scheduler']} = ReduceLROnPlateau({opt}, mode={mode}, factor={factor}, patience={patience})"
        ]
