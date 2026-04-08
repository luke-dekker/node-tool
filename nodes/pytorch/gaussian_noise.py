"""Gaussian Noise node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class GaussianNoiseNode(BaseNode):
    type_name   = "pt_gaussian_noise"
    label       = "Gaussian Noise"
    category    = "Models"
    subcategory = "Autoencoder"
    description = "Add Gaussian noise to a tensor. Useful for denoising autoencoders: corrupt input, reconstruct clean."

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("std",    PortType.FLOAT,  default=0.1)
        self.add_input("clip",   PortType.BOOL,   default=True)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            t   = inputs.get("tensor")
            std = float(inputs.get("std") or 0.1)
            if t is None:
                return {"tensor": None}
            noisy = t + torch.randn_like(t) * std
            if bool(inputs.get("clip", True)):
                noisy = noisy.clamp(0.0, 1.0)
            return {"tensor": noisy}
        except Exception:
            return {"tensor": None}

    def export(self, iv, ov):
        t = iv.get("tensor") or "None  # TODO: connect input tensor"
        out = ov.get("tensor", "_noisy")
        std = self._val(iv, "std")
        clip = bool(self.inputs["clip"].default_value)
        lines = [f"{out} = {t} + torch.randn_like({t}) * {std}"]
        if clip:
            lines.append(f"{out} = {out}.clamp(0.0, 1.0)")
        return ["import torch"], lines
