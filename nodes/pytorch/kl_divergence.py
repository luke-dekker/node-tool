"""KL Divergence node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class KLDivergenceNode(BaseNode):
    type_name   = "pt_kl_divergence"
    label       = "KL Divergence"
    category    = "Models"
    subcategory = "Autoencoder"
    description = "KL divergence loss for VAE: -0.5 * sum(1 + log_var - mu^2 - exp(log_var)). Returns scalar mean over batch."

    def _setup_ports(self):
        self.add_input("mu",      PortType.TENSOR, default=None)
        self.add_input("log_var", PortType.TENSOR, default=None)
        self.add_output("kl_loss", PortType.TENSOR)
        self.add_output("info",    PortType.STRING)

    def execute(self, inputs):
        try:
            import torch
            mu      = inputs.get("mu")
            log_var = inputs.get("log_var")
            if mu is None or log_var is None:
                return {"kl_loss": None, "info": "Need mu and log_var"}
            kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            return {"kl_loss": kl, "info": f"KL={kl.item():.6f}"}
        except Exception:
            return {"kl_loss": None, "info": "error"}

    def export(self, iv, ov):
        mu = iv.get("mu") or "None  # TODO: connect mu tensor"
        lv = iv.get("log_var") or "None  # TODO: connect log_var tensor"
        kl_var = ov.get("kl_loss", "_kl")
        info_var = ov.get("info",  "_kl_info")
        return ["import torch"], [
            f"{kl_var} = -0.5 * torch.mean(1 + {lv} - {mu}.pow(2) - {lv}.exp())",
            f"{info_var} = f'KL={{{kl_var}.item():.6f}}'",
        ]
