"""Load Weights node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class LoadWeightsNode(BaseNode):
    """Load state_dict into an existing model architecture."""
    type_name   = "pt_load_weights"
    label       = "Load Weights"
    category    = "Models"
    subcategory = "Save & Load"
    description = (
        "Load saved weights into a model. "
        "The architecture must match the saved file."
    )

    def _setup_ports(self) -> None:
        self.add_input("model",  PortType.MODULE, default=None,
                       description="Model with matching architecture")
        self.add_input("path",   PortType.STRING, default="model_weights.pt")
        self.add_input("device", PortType.STRING, default="cpu",
                       description="Device to map weights to (cpu / cuda)")
        self.add_output("model", PortType.MODULE, description="Model with loaded weights")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        model  = inputs.get("model")
        path   = inputs.get("path") or "model_weights.pt"
        device = inputs.get("device") or "cpu"
        if model is not None:
            state = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.to(device)
        return {"model": model}

    def export(self, iv, ov):
        model  = iv.get("model", "None")
        path   = iv.get("path", repr(self.inputs["path"].default_value))
        device = iv.get("device", repr(self.inputs["device"].default_value))
        out    = ov.get("model", "_model")
        lines  = [
            f"_ckpt_state = torch.load({path}, map_location={device}, weights_only=True)",
            f"{model}.load_state_dict(_ckpt_state)",
            f"{model}.to({device})",
            f"{out} = {model}",
        ]
        return ["import torch"], lines
