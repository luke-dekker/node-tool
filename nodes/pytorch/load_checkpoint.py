"""Load Checkpoint node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class LoadCheckpointNode(BaseNode):
    """Load a training checkpoint — restore model, optimizer, and epoch."""
    type_name   = "pt_load_checkpoint"
    label       = "Load Checkpoint"
    category    = "Models"
    subcategory = "Save & Load"
    description = (
        "Restore model and optimizer from a checkpoint file. "
        "Resume training from the saved epoch."
    )

    def _setup_ports(self) -> None:
        self.add_input("model",      PortType.MODULE,    default=None,
                       description="Model with matching architecture")
        self.add_input("optimizer",  PortType.OPTIMIZER, default=None)
        self.add_input("path",       PortType.STRING,    default="checkpoint.pt")
        self.add_input("device",     PortType.STRING,    default="cpu")
        self.add_output("model",     PortType.MODULE,    description="Model with restored weights")
        self.add_output("optimizer", PortType.OPTIMIZER, description="Optimizer with restored state")
        self.add_output("epoch",     PortType.INT,       description="Epoch checkpoint was saved at")
        self.add_output("loss",      PortType.FLOAT,     description="Loss at checkpoint time")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        model     = inputs.get("model")
        optimizer = inputs.get("optimizer")
        path      = inputs.get("path") or "checkpoint.pt"
        device    = inputs.get("device") or "cpu"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if model is not None and "model_state_dict" in ckpt and ckpt["model_state_dict"] is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
        if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return {
            "model":     model,
            "optimizer": optimizer,
            "epoch":     ckpt.get("epoch", 0),
            "loss":      ckpt.get("loss",  0.0),
        }

    def export(self, iv, ov):
        model     = iv.get("model", "None")
        optimizer = iv.get("optimizer", "None")
        path      = iv.get("path", repr(self.inputs["path"].default_value))
        device    = iv.get("device", repr(self.inputs["device"].default_value))
        out_model = ov.get("model", "_model")
        out_opt   = ov.get("optimizer", "_optimizer")
        out_epoch = ov.get("epoch", "_epoch")
        out_loss  = ov.get("loss", "_loss")
        lines = [
            f"_ckpt = torch.load({path}, map_location={device}, weights_only=False)",
            f"if {model} is not None and 'model_state_dict' in _ckpt:",
            f"    {model}.load_state_dict(_ckpt['model_state_dict'])",
            f"    {model}.to({device})",
            f"if {optimizer} is not None and 'optimizer_state_dict' in _ckpt:",
            f"    {optimizer}.load_state_dict(_ckpt['optimizer_state_dict'])",
            f"{out_model} = {model}",
            f"{out_opt} = {optimizer}",
            f"{out_epoch} = _ckpt.get('epoch', 0)",
            f"{out_loss} = _ckpt.get('loss', 0.0)",
        ]
        return ["import torch"], lines
