"""Save Full Model node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class SaveFullModelNode(BaseNode):
    """Save the entire model object (architecture + weights) to a .pt file."""
    type_name   = "pt_save_full_model"
    label       = "Save Full Model"
    category    = "Models"
    subcategory = "Save & Load"
    description = (
        "torch.save(model, path) — saves architecture AND weights together. "
        "Load back with Pretrained Block on any canvas, no graph rebuilding needed."
    )

    def _setup_ports(self) -> None:
        self.add_input("model",  PortType.MODULE, default=None)
        self.add_input("path",   PortType.STRING, default="model_full.pt")
        self.add_output("model", PortType.MODULE, description="Pass-through model")
        self.add_output("info",  PortType.STRING, description="Save confirmation")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        model = inputs.get("model")
        path  = inputs.get("path") or "model_full.pt"
        if model is None:
            return {"model": None, "info": "No model connected."}
        try:
            torch.save(model, path)
            total = sum(p.numel() for p in model.parameters())
            info  = f"Saved {model.__class__.__name__} ({total:,} params) → {path}"
        except Exception as exc:
            info = f"Save failed: {exc}"
        return {"model": model, "info": info}

    def export(self, iv, ov):
        model = iv.get("model", "None")
        path  = self._val(iv, "path")
        out_m = ov.get("model", "_model_pass")
        lines = [
            f"torch.save({model}, {path})",
            f"{out_m} = {model}",
        ]
        return ["import torch"], lines
