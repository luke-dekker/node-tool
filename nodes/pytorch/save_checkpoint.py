"""Save Checkpoint node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class SaveCheckpointNode(BaseNode):
    """Save a full training checkpoint (model + optimizer + epoch)."""
    type_name   = "pt_save_checkpoint"
    label       = "Save Checkpoint"
    category    = "Models"
    subcategory = "Save & Load"
    description = (
        "Save model weights, optimizer state, and current epoch. "
        "Use LoadCheckpointNode to resume training exactly."
    )

    def _setup_ports(self) -> None:
        self.add_input("model",     PortType.MODULE,    default=None)
        self.add_input("optimizer", PortType.OPTIMIZER, default=None)
        self.add_input("epoch",     PortType.INT,       default=0)
        self.add_input("loss",      PortType.FLOAT,     default=0.0)
        self.add_input("path",      PortType.STRING,    default="checkpoint.pt")
        self.add_output("path",     PortType.STRING,    description="Path that was saved to")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        model     = inputs.get("model")
        optimizer = inputs.get("optimizer")
        path      = inputs.get("path") or "checkpoint.pt"
        ckpt: dict[str, Any] = {
            "epoch": inputs.get("epoch", 0),
            "loss":  inputs.get("loss",  0.0),
            "model_state_dict":     model.state_dict()     if model     is not None else None,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        }
        torch.save(ckpt, path)
        return {"path": path}

    def export(self, iv, ov):
        model     = iv.get("model", "None")
        optimizer = iv.get("optimizer", "None")
        epoch     = iv.get("epoch", "0")
        loss      = iv.get("loss", "0.0")
        path      = iv.get("path", repr(self.inputs["path"].default_value))
        lines = [
            f"torch.save({{",
            f'    "epoch": {epoch},',
            f'    "loss": {loss},',
            f'    "model_state_dict": {model}.state_dict() if {model} is not None else None,',
            f'    "optimizer_state_dict": {optimizer}.state_dict() if {optimizer} is not None else None,',
            f"}}, {path})",
        ]
        out_path = ov.get("path")
        if out_path:
            lines.append(f"{out_path} = {path}")
        return ["import torch"], lines
