"""Model Info node — prints model summary + param counts."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class ModelInfoNode(BaseNode):
    """Display model architecture summary and parameter count."""
    type_name   = "pt_model_info"
    label       = "Model Info"
    category    = "Models"
    subcategory = "Inspect"
    description = "Print model architecture and total parameter count."

    def _setup_ports(self) -> None:
        self.add_input("model",  PortType.MODULE, default=None)
        self.add_output("model", PortType.MODULE, description="Pass-through model")
        self.add_output("info",  PortType.STRING, description="Architecture summary")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        model = inputs.get("model")
        if model is None:
            return {"model": None, "info": "No model connected."}
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lines = [
            str(model),
            f"\nTotal params:     {total:,}",
            f"Trainable params: {trainable:,}",
            f"Frozen params:    {total - trainable:,}",
        ]
        return {"model": model, "info": "\n".join(lines)}

    def export(self, iv, ov):
        model    = iv.get("model", "None")
        out_m    = ov.get("model", "_model_passthrough")
        out_info = ov.get("info", "_model_info")
        lines = [
            f"_total_p = sum(p.numel() for p in {model}.parameters())",
            f"_train_p = sum(p.numel() for p in {model}.parameters() if p.requires_grad)",
            f"{out_info} = f'Params: {{_total_p:,}} total / {{_train_p:,}} trainable'",
            f"print({out_info})",
            f"{out_m} = {model}",
        ]
        return [], lines
