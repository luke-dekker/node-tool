"""Reparameterize node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ReparameterizeNode(BaseNode):
    type_name   = "pt_reparameterize"
    label       = "Reparameterize"
    category    = "Models"
    subcategory = "Autoencoder"
    description = "Reparameterization trick: z = mu + eps * exp(0.5 * log_var). Use inside VAE forward pass for differentiable sampling."

    def _setup_ports(self):
        self.add_input("mu",      PortType.TENSOR, default=None)
        self.add_input("log_var", PortType.TENSOR, default=None)
        self.add_output("z", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            mu      = inputs.get("mu")
            log_var = inputs.get("log_var")
            if mu is None or log_var is None:
                return {"z": None}
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return {"z": mu + eps * std}
        except Exception:
            return {"z": None}
