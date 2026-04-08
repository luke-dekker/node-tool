"""Latent Sampler node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class LatentSamplerNode(BaseNode):
    type_name   = "pt_latent_sampler"
    label       = "Latent Sampler"
    category    = "Models"
    subcategory = "Autoencoder"
    description = "Sample n points from N(0,1) in the latent space and decode them through a VAE decoder module."

    def _setup_ports(self):
        self.add_input("decoder",    PortType.MODULE, default=None)
        self.add_input("latent_dim", PortType.INT,    default=16)
        self.add_input("n_samples",  PortType.INT,    default=8)
        self.add_input("device",     PortType.STRING, default="cpu")
        self.add_output("samples", PortType.TENSOR)
        self.add_output("info",    PortType.STRING)

    def execute(self, inputs):
        try:
            import torch
            decoder    = inputs.get("decoder")
            latent_dim = int(inputs.get("latent_dim") or 16)
            n          = int(inputs.get("n_samples") or 8)
            device     = str(inputs.get("device") or "cpu")
            if decoder is None:
                return {"samples": None, "info": "No decoder"}
            z = torch.randn(n, latent_dim).to(device)
            with torch.no_grad():
                samples = decoder(z)
            return {"samples": samples, "info": f"Sampled {n} points from N(0,I) in R^{latent_dim}"}
        except Exception:
            import traceback
            return {"samples": None, "info": traceback.format_exc().split("\n")[-2]}

    def export(self, iv, ov):
        decoder = iv.get("decoder") or "None  # TODO: connect a decoder module"
        latent_dim = self._val(iv, "latent_dim")
        n = self._val(iv, "n_samples")
        device = self._val(iv, "device")
        s_var = ov.get("samples", "_samples")
        i_var = ov.get("info",    "_samples_info")
        return ["import torch"], [
            f"_z = torch.randn({n}, {latent_dim}).to({device})",
            f"with torch.no_grad():",
            f"    {s_var} = {decoder}(_z)",
            f"{i_var} = f'Sampled {n} points from N(0,I) in R^{latent_dim}'",
        ]
