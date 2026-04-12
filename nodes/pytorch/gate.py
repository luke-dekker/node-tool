"""GateNode — per-layer training experiment control.

Wire between any two layers to control what flows during training. Four modes:

    pass    — normal passthrough (default, no effect)
    zeros   — replace the tensor with zeros of the same shape (ablation)
    noise   — add Gaussian noise (corruption / regularization)
    detach  — pass the tensor but block gradients (like a selective freeze
              at the activation level, not the parameter level)

Useful for:
    - Ablation studies: set gate to 'zeros' to test how downstream layers
      perform without an encoder's signal
    - Noise injection: set gate to 'noise' to corrupt representations and
      measure robustness or train a denoising objective
    - Gradient isolation: set gate to 'detach' to train a downstream head
      without updating the upstream encoder (similar to freezing the encoder
      but without touching its parameters — useful when the encoder is
      shared across tasks and you want to freeze it for ONE task but not others)
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


_MODES = ["pass", "zeros", "noise", "detach"]


class GateNode(BaseNode):
    type_name   = "pt_gate"
    label       = "Gate"
    category    = "Layers"
    subcategory = "Utility"
    description = (
        "Training experiment control. Modes: pass (normal), zeros (ablation), "
        "noise (corruption), detach (block gradients). Wire between layers "
        "to control what flows during training."
    )

    def _setup_ports(self) -> None:
        self.add_input("tensor_in", PortType.TENSOR, default=None)
        self.add_input("mode",      PortType.STRING, default="pass",
                       choices=_MODES,
                       description="pass=normal, zeros=ablation, noise=corruption, "
                                   "detach=block gradients")
        self.add_input("noise_std", PortType.FLOAT,  default=0.1,
                       description="Noise standard deviation (only used in noise mode)")
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        t = inputs.get("tensor_in")
        if t is None:
            return {"tensor_out": None}

        mode = str(inputs.get("mode") or "pass").strip().lower()
        try:
            import torch
            if mode == "zeros":
                return {"tensor_out": torch.zeros_like(t)}
            if mode == "noise":
                std = float(inputs.get("noise_std") or 0.1)
                return {"tensor_out": t + torch.randn_like(t) * std}
            if mode == "detach":
                return {"tensor_out": t.detach()}
            # default: pass
            return {"tensor_out": t}
        except Exception:
            return {"tensor_out": t}

    def export(self, iv, ov):
        tin  = iv.get("tensor_in") or "None"
        mode = str(self.inputs["mode"].default_value or "pass").strip().lower()
        std  = float(self.inputs["noise_std"].default_value or 0.1)
        tout = ov.get("tensor_out", "_gated")
        if mode == "zeros":
            return ["import torch"], [f"{tout} = torch.zeros_like({tin})"]
        if mode == "noise":
            return ["import torch"], [f"{tout} = {tin} + torch.randn_like({tin}) * {std}"]
        if mode == "detach":
            return ["import torch"], [f"{tout} = {tin}.detach()"]
        return [], [f"{tout} = {tin}"]
