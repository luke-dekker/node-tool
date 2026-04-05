"""PyTorch training/inference nodes."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Training"


class ForwardPassNode(BaseNode):
    type_name = "pt_forward_pass"
    label = "Forward Pass"
    category = CATEGORY
    description = "Run model.eval() forward pass on input tensor"

    def _setup_ports(self):
        self.add_input("model", PortType.MODULE, default=None)
        self.add_input("input", PortType.TENSOR, default=None)
        self.add_output("output", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            model = inputs.get("model")
            inp = inputs.get("input")
            if model is None or inp is None:
                return {"output": None}
            model.eval()
            with torch.no_grad():
                out = model(inp)
            return {"output": out}
        except Exception:
            return {"output": None}


_OPT_HELP  = "adam | adamw | sgd | rmsprop"
_LOSS_HELP = "crossentropy | mse | bce | bcewithlogits | l1"
_SCH_HELP  = "none | steplr | cosine | reducelr"


def _build_optimizer(name: str, model, lr: float, weight_decay: float, momentum: float):
    """Construct a torch optimizer from a string name."""
    import torch.optim as optim
    if model is None:
        return None
    params = model.parameters()
    key = name.strip().lower().replace("_", "").replace(" ", "")
    if key == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    if key == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)  # default: adam


def _build_loss(name: str):
    """Construct a torch loss function from a string name."""
    import torch.nn as nn
    key = name.strip().lower().replace("_", "").replace(" ", "").replace("-", "")
    return {
        "mse":            nn.MSELoss(),
        "bce":            nn.BCELoss(),
        "bcewithlogits":  nn.BCEWithLogitsLoss(),
        "l1":             nn.L1Loss(),
    }.get(key, nn.CrossEntropyLoss())  # default: crossentropy


def _build_scheduler(name: str, optimizer, step_size: int, gamma: float, T_max: int):
    """Construct a torch scheduler from a string name."""
    if optimizer is None:
        return None
    key = name.strip().lower().replace("_", "").replace(" ", "")
    if key == "steplr":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if key == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=T_max)
    if key == "reducelr":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer, factor=gamma, patience=step_size)
    return None  # "none"


class TrainingConfigNode(BaseNode):
    type_name = "pt_training_config"
    label = "Training Config"
    category = CATEGORY
    description = (
        "Wire the last layer's tensor_out into tensor_in, and dataloaders below. "
        f"Optimizer: {_OPT_HELP}  |  Loss: {_LOSS_HELP}  |  Scheduler: {_SCH_HELP}"
    )

    def _setup_ports(self):
        # tensor_in receives predictions from the last layer
        self.add_input("tensor_in",      PortType.TENSOR,     default=None,
                       description="Wire from the last layer's tensor_out")
        self.add_input("dataloader",     PortType.DATALOADER, default=None)
        self.add_input("val_dataloader", PortType.DATALOADER, default=None,
                       description="Optional validation dataloader")
        # Training hyperparams
        self.add_input("epochs",         PortType.INT,        default=10)
        self.add_input("device",         PortType.STRING,     default="cpu",
                       description="cpu | cuda | cuda:0")
        # Optimizer
        self.add_input("optimizer",      PortType.STRING,     default="adam",
                       description=_OPT_HELP,
                       choices=["adam", "adamw", "sgd", "rmsprop"])
        self.add_input("lr",             PortType.FLOAT,      default=0.001)
        self.add_input("weight_decay",   PortType.FLOAT,      default=0.0)
        self.add_input("momentum",       PortType.FLOAT,      default=0.9,
                       description="SGD only")
        # Loss
        self.add_input("loss",           PortType.STRING,     default="crossentropy",
                       description=_LOSS_HELP,
                       choices=["crossentropy", "mse", "bce", "bcewithlogits", "l1"])
        # Scheduler
        self.add_input("scheduler",      PortType.STRING,     default="none",
                       description=_SCH_HELP,
                       choices=["none", "steplr", "cosine", "reducelr"])
        self.add_input("step_size",      PortType.INT,        default=10,
                       description="StepLR: step size  |  ReduceLR: patience")
        self.add_input("gamma",          PortType.FLOAT,      default=0.1,
                       description="StepLR / ReduceLR decay factor")
        self.add_input("T_max",          PortType.INT,        default=50,
                       description="CosineAnnealing: T_max")
        self.add_output("config",        PortType.ANY)

    def execute(self, inputs):
        try:
            # Return hyperparams only — model/optimizer built in app._start_training()
            # by traversing the tensor_in connection chain.
            loss_fn = _build_loss(inputs.get("loss") or "crossentropy")
            return {"config": {
                "loss_fn":        loss_fn,
                "dataloader":     inputs.get("dataloader"),
                "val_dataloader": inputs.get("val_dataloader"),
                "epochs":         int(inputs.get("epochs")       or 10),
                "device":         str(inputs.get("device")       or "cpu"),
                "optimizer_name": str(inputs.get("optimizer")    or "adam"),
                "lr":             float(inputs.get("lr")         or 0.001),
                "weight_decay":   float(inputs.get("weight_decay") or 0.0),
                "momentum":       float(inputs.get("momentum")   or 0.9),
                "scheduler_name": str(inputs.get("scheduler")    or "none"),
                "step_size":      int(inputs.get("step_size")    or 10),
                "gamma":          float(inputs.get("gamma")      or 0.1),
                "T_max":          int(inputs.get("T_max")        or 50),
            }}
        except Exception:
            return {"config": None}


# Subcategory stamp
_SC = "Config"
from core.node import BaseNode as _BN
for _n, _c in list(globals().items()):
    if isinstance(_c, type) and issubclass(_c, _BN) and _c is not _BN:
        _c.subcategory = _SC
