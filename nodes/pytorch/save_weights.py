"""Save Weights node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class SaveWeightsNode(BaseNode):
    """Save model state_dict (weights only) to a .pt file."""
    type_name   = "pt_save_weights"
    label       = "Save Weights"
    category    = "Models"
    subcategory = "Save & Load"
    description = "Save model.state_dict() to a file. Portable across architectures."

    def _setup_ports(self) -> None:
        self.add_input("model",  PortType.MODULE, default=None)
        self.add_input("path",   PortType.STRING, default="model_weights.pt")
        self.add_output("model", PortType.MODULE, description="Pass-through model")
        self.add_output("path",  PortType.STRING, description="Path that was saved to")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        model = inputs.get("model")
        path  = inputs.get("path") or "model_weights.pt"
        if model is not None:
            torch.save(model.state_dict(), path)
        return {"model": model, "path": path}

    def export(self, iv, ov):
        model = iv.get("model", "None")
        path  = iv.get("path", repr(self.inputs["path"].default_value))
        lines = [
            f"if {model} is not None:",
            f"    torch.save({model}.state_dict(), {path})",
        ]
        if "model" in ov:
            lines.append(f"{ov['model']} = {model}")
        if "path" in ov:
            lines.append(f"{ov['path']} = {path}")
        return ["import torch"], lines
