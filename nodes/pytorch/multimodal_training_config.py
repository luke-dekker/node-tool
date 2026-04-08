"""MultimodalTrainingConfig — training config for multimodal models with selective freezing."""
from __future__ import annotations
from core.node import BaseNode, PortType
from nodes.pytorch.training_config import _build_optimizer, _build_loss, _build_scheduler


_FREEZE_HELP = (
    "hard: freeze inactive encoder weights (requires_grad=False)  |  "
    "noise: corrupt inactive branch outputs with noise  |  "
    "embed: replace inactive branch outputs with learnable mask token  |  "
    "none: train all branches every batch"
)


class MultimodalTrainingConfigNode(BaseNode):
    type_name   = "pt_multimodal_training_config"
    label       = "Multimodal Training Config"
    category    = "Training"
    subcategory = "Config"
    description = (
        "Training config for multimodal models. Per batch, detect which modalities are "
        "present and apply the freeze strategy to inactive encoder branches. "
        f"Freeze strategies: {_FREEZE_HELP}"
    )

    def _setup_ports(self):
        # Wire in the multimodal model's tensor_out (so we can find the model upstream)
        self.add_input("tensor_in",      PortType.TENSOR,     default=None,
                       description="Wire from MultimodalModel's tensor_out")
        self.add_input("dataloader",     PortType.DATALOADER, default=None,
                       description="Multi Dataset loader (or any multimodal-collated DataLoader)")
        self.add_input("val_dataloader", PortType.DATALOADER, default=None)

        # Multimodal-specific
        self.add_input("freeze_strategy", PortType.STRING, default="hard",
                       choices=["hard", "noise", "embed", "none"],
                       description=_FREEZE_HELP)
        self.add_input("log_modalities",  PortType.BOOL,   default=True,
                       description="Log which modalities trained per batch")

        # Standard hyperparams
        self.add_input("epochs",       PortType.INT,    default=10)
        self.add_input("device",       PortType.STRING, default="cpu",
                       description="cpu | cuda | cuda:0")
        self.add_input("optimizer",    PortType.STRING, default="adam",
                       choices=["adam", "adamw", "sgd", "rmsprop"])
        self.add_input("lr",           PortType.FLOAT,  default=0.001)
        self.add_input("weight_decay", PortType.FLOAT,  default=0.0)
        self.add_input("momentum",     PortType.FLOAT,  default=0.9)
        self.add_input("loss",         PortType.STRING, default="crossentropy",
                       choices=["crossentropy", "mse", "bce", "bcewithlogits", "l1"])
        self.add_input("scheduler",    PortType.STRING, default="none",
                       choices=["none", "steplr", "cosine", "reducelr"])
        self.add_input("step_size",    PortType.INT,    default=10)
        self.add_input("gamma",        PortType.FLOAT,  default=0.1)
        self.add_input("T_max",        PortType.INT,    default=50)

        self.add_output("config", PortType.ANY)

    def execute(self, inputs):
        try:
            return {"config": {
                "multimodal":      True,
                "loss_fn":         _build_loss(inputs.get("loss") or "crossentropy"),
                "dataloader":      inputs.get("dataloader"),
                "val_dataloader":  inputs.get("val_dataloader"),
                "epochs":          int(inputs.get("epochs")       or 10),
                "device":          str(inputs.get("device")       or "cpu"),
                "optimizer_name":  str(inputs.get("optimizer")    or "adam"),
                "lr":              float(inputs.get("lr")         or 0.001),
                "weight_decay":    float(inputs.get("weight_decay") or 0.0),
                "momentum":        float(inputs.get("momentum")   or 0.9),
                "scheduler_name":  str(inputs.get("scheduler")    or "none"),
                "step_size":       int(inputs.get("step_size")    or 10),
                "gamma":           float(inputs.get("gamma")      or 0.1),
                "T_max":           int(inputs.get("T_max")        or 50),
                "freeze_strategy": str(inputs.get("freeze_strategy") or "hard"),
                "log_modalities":  bool(inputs.get("log_modalities", True)),
            }}
        except Exception:
            return {"config": None}

    def export(self, iv, ov):
        return [], [
            f"# {self.label}: multimodal training requires the GUI training loop.",
            f"# Code export is not yet supported. Use the GUI to train, then save a checkpoint.",
        ]
