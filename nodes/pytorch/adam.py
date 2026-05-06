"""Optimizer factory node — replaces AdamNode, AdamWNode, and SGDNode with a single
dropdown-driven node. Also supports RMSprop."""
from __future__ import annotations
from core.node import BaseNode, PortType


class OptimizerNode(BaseNode):
    type_name   = "pt_optimizer"
    label       = "Optimizer"
    category    = "Training"
    subcategory = "Optimizers"
    description = "Construct a PyTorch optimizer selected by dropdown."

    def _setup_ports(self):
        self.add_input("model",          PortType.MODULE,  default=None)
        self.add_input("optimizer_type", PortType.STRING,  default="adam",
                       choices=["adam", "adamw", "sgd", "rmsprop"])
        self.add_input("lr",             PortType.FLOAT,   default=0.001)
        self.add_input("weight_decay",   PortType.FLOAT,   default=0.0)
        self.add_input("momentum",       PortType.FLOAT,   default=0.9)
        self.add_output("optimizer", PortType.OPTIMIZER)

    def execute(self, inputs):
        try:
            model = inputs.get("model")
            if model is None:
                return {"optimizer": None}
            import torch.optim as optim
            opt_type = str(inputs.get("optimizer_type", "adam") or "adam")
            lr = float(inputs.get("lr", 0.001))
            wd = float(inputs.get("weight_decay", 0.0))
            mom = float(inputs.get("momentum", 0.9))
            params = model.parameters()
            if opt_type == "adam":
                return {"optimizer": optim.Adam(params, lr=lr, weight_decay=wd)}
            elif opt_type == "adamw":
                return {"optimizer": optim.AdamW(params, lr=lr, weight_decay=wd)}
            elif opt_type == "sgd":
                return {"optimizer": optim.SGD(params, lr=lr, momentum=mom, weight_decay=wd)}
            elif opt_type == "rmsprop":
                return {"optimizer": optim.RMSprop(params, lr=lr, momentum=mom, weight_decay=wd)}
            return {"optimizer": None}
        except Exception:
            return {"optimizer": None}

    def export(self, iv, ov):
        m = self._val(iv, 'model')
        opt_type = (self.inputs["optimizer_type"].default_value or "adam")
        lr = self._val(iv, 'lr')
        wd = self._val(iv, 'weight_decay')
        mom = self._val(iv, 'momentum')
        cls_map = {
            "adam":    "Adam",
            "adamw":   "AdamW",
            "sgd":     "SGD",
            "rmsprop": "RMSprop",
        }
        cls = cls_map.get(opt_type, "Adam")
        if opt_type in ("sgd", "rmsprop"):
            args = f"{m}.parameters(), lr={lr}, momentum={mom}, weight_decay={wd}"
        else:
            args = f"{m}.parameters(), lr={lr}, weight_decay={wd}"
        return ["import torch.optim as optim"], [
            f"{ov['optimizer']} = optim.{cls}({args})"
        ]
