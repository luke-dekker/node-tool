"""Training Config node."""
from __future__ import annotations
from core.node import BaseNode, PortType

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
    type_name   = "pt_training_config"
    label       = "Training Config"
    category    = "Training"
    subcategory = "Config"
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

    def export(self, iv, ov):
        model      = iv.get('tensor_in') or '_model'
        loader     = self._val(iv, 'dataloader')
        val_loader = iv.get('val_dataloader')
        epochs     = self._val(iv, 'epochs')
        device     = self._val(iv, 'device')

        opt_name  = self.inputs["optimizer"].default_value if "optimizer" in self.inputs else "adam"
        lr        = self.inputs["lr"].default_value        if "lr"        in self.inputs else 0.001
        wd        = self.inputs["weight_decay"].default_value if "weight_decay" in self.inputs else 0.0
        mom       = self.inputs["momentum"].default_value  if "momentum"  in self.inputs else 0.9
        loss_name = self.inputs["loss"].default_value      if "loss"      in self.inputs else "crossentropy"
        sch_name  = self.inputs["scheduler"].default_value if "scheduler" in self.inputs else "none"
        step_size = self.inputs["step_size"].default_value if "step_size" in self.inputs else 10
        gamma     = self.inputs["gamma"].default_value     if "gamma"     in self.inputs else 0.1
        T_max     = self.inputs["T_max"].default_value     if "T_max"     in self.inputs else 50

        opt_map = {
            "adamw":   f"torch.optim.AdamW({model}.parameters(), lr={lr}, weight_decay={wd})",
            "sgd":     f"torch.optim.SGD({model}.parameters(), lr={lr}, weight_decay={wd}, momentum={mom})",
            "rmsprop": f"torch.optim.RMSprop({model}.parameters(), lr={lr}, weight_decay={wd})",
        }
        opt_expr = opt_map.get(str(opt_name).strip().lower(),
                               f"torch.optim.Adam({model}.parameters(), lr={lr}, weight_decay={wd})")

        loss_map = {
            "mse":           "torch.nn.MSELoss()",
            "bce":           "torch.nn.BCELoss()",
            "bcewithlogits": "torch.nn.BCEWithLogitsLoss()",
            "l1":            "torch.nn.L1Loss()",
        }
        loss_expr = loss_map.get(str(loss_name).strip().lower().replace("-", ""),
                                 "torch.nn.CrossEntropyLoss()")

        sch_key = str(sch_name).strip().lower().replace("_", "")
        if sch_key == "steplr":
            sch_expr = f"torch.optim.lr_scheduler.StepLR(_optimizer, step_size={step_size}, gamma={gamma})"
        elif sch_key == "cosine":
            sch_expr = f"torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max={T_max})"
        elif sch_key == "reducelr":
            sch_expr = f"torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizer, factor={gamma}, patience={step_size})"
        else:
            sch_expr = None

        lines = [
            "# Training setup",
            f"_device   = torch.device({device})",
            f"{model}   = {model}.to(_device)",
            f"_optimizer = {opt_expr}",
            f"_loss_fn   = {loss_expr}",
        ]
        if sch_expr:
            lines.append(f"_scheduler = {sch_expr}")

        lines += [
            f"# Training loop",
            f"for _epoch in range(1, int({epochs}) + 1):",
            f"    {model}.train()",
            f"    _train_loss = 0.0",
            f"    for _batch in {loader}:",
            f"        if isinstance(_batch, (list, tuple)):",
            f"            _x, _y = _batch[0].to(_device), _batch[1].to(_device)",
            f"        else:",
            f"            _x = _batch.to(_device); _y = _x",
            f"        _optimizer.zero_grad()",
            f"        _out = {model}(_x)",
            f"        _loss = _loss_fn(_out, _y)",
            f"        _loss.backward()",
            f"        _optimizer.step()",
            f"        _train_loss += _loss.item()",
            f"    _train_loss /= len({loader})",
        ]

        if val_loader is not None:
            lines += [
                f"    # Validation",
                f"    {model}.eval()",
                f"    _val_loss = 0.0",
                f"    with torch.no_grad():",
                f"        for _vbatch in {val_loader}:",
                f"            if isinstance(_vbatch, (list, tuple)):",
                f"                _vx, _vy = _vbatch[0].to(_device), _vbatch[1].to(_device)",
                f"            else:",
                f"                _vx = _vbatch.to(_device); _vy = _vx",
                f"            _val_loss += _loss_fn({model}(_vx), _vy).item()",
                f"    _val_loss /= len({val_loader})",
            ]

        if sch_expr:
            lines += [
                f"    # LR scheduler step",
                f"    from torch.optim.lr_scheduler import ReduceLROnPlateau as _RLRP",
                f"    if isinstance(_scheduler, _RLRP):",
                f"        _scheduler.step({'_val_loss' if val_loader is not None else '_train_loss'})",
                f"    else:",
                f"        _scheduler.step()",
            ]

        if val_loader is not None:
            lines.append(
                f"    print(f\"Epoch {{_epoch}}/{{{epochs}}}  train={{_train_loss:.6f}}  val={{_val_loss:.6f}}\")"
            )
        else:
            lines.append(
                f"    print(f\"Epoch {{_epoch}}/{{{epochs}}}  loss={{_train_loss:.6f}}\")"
            )

        return ["import torch"], lines
