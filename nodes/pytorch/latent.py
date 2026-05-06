"""Consolidated VAE latent helper — replaces ReparameterizeNode + LatentSamplerNode.

Pick `mode`:
  reparameterize — z = mean + eps * exp(0.5 * log_variance) (encoder→decoder)
  sample         — torch.randn(n, latent_dim) → decoder(z) (after-training sampling)
"""
from __future__ import annotations
from core.node import BaseNode, PortType


_MODES = ["reparameterize", "sample"]


class LatentNode(BaseNode):
    type_name   = "pt_latent"
    label       = "Latent"
    category    = "Models"
    subcategory = "Autoencoder"
    description = (
        "Pick `mode`:\n"
        "  reparameterize — VAE training: combine `mean`, `log_variance` → `z`\n"
        "  sample         — N samples from N(0,I) decoded through `decoder` → `samples`"
    )

    def relevant_inputs(self, values):
        mode = (values.get("mode") or "reparameterize").strip()
        if mode == "reparameterize": return ["mode"]   # mean + log_variance wired
        if mode == "sample":         return ["mode", "latent_dim", "n_samples", "device"]
        return ["mode"]

    def _setup_ports(self):
        self.add_input("mode",         PortType.STRING, "reparameterize", choices=_MODES)
        # reparameterize inputs
        self.add_input("mean",         PortType.TENSOR, default=None, optional=True)
        self.add_input("log_variance", PortType.TENSOR, default=None, optional=True)
        # sample inputs
        self.add_input("decoder",      PortType.MODULE, default=None, optional=True)
        self.add_input("latent_dim",   PortType.INT,    16,           optional=True)
        self.add_input("n_samples",    PortType.INT,    8,            optional=True)
        self.add_input("device",       PortType.STRING, "cpu",        optional=True)
        self.add_output("z",       PortType.TENSOR)
        self.add_output("samples", PortType.TENSOR)
        self.add_output("info",    PortType.STRING)

    def execute(self, inputs):
        out = {"z": None, "samples": None, "info": ""}
        try:
            import torch
            mode = (inputs.get("mode") or "reparameterize").strip()
            if mode == "reparameterize":
                mean    = inputs.get("mean")
                log_var = inputs.get("log_variance")
                if mean is None or log_var is None:
                    out["info"] = "Need mean and log_variance"; return out
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                out["z"] = mean + eps * std
                return out
            if mode == "sample":
                decoder = inputs.get("decoder")
                if decoder is None:
                    out["info"] = "No decoder"; return out
                latent_dim = int(inputs.get("latent_dim") or 16)
                n          = int(inputs.get("n_samples")  or 8)
                device     = str(inputs.get("device") or "cpu")
                z = torch.randn(n, latent_dim).to(device)
                with torch.no_grad():
                    out["samples"] = decoder(z)
                out["info"] = f"Sampled {n} points from N(0,I) in R^{latent_dim}"
                return out
            return out
        except Exception as exc:
            return {"z": None, "samples": None, "info": f"error: {exc}"}

    def export(self, iv, ov):
        mode = (self.inputs["mode"].default_value or "reparameterize")
        if mode == "reparameterize":
            mu = iv.get("mean") or "None"; lv = iv.get("log_variance") or "None"
            z  = ov.get("z", "_z")
            return ["import torch"], [
                f"_std = torch.exp(0.5 * {lv})",
                f"_eps = torch.randn_like(_std)",
                f"{z} = {mu} + _eps * _std",
            ]
        if mode == "sample":
            decoder = iv.get("decoder") or "None"
            ld = self._val(iv, "latent_dim"); n = self._val(iv, "n_samples"); dev = self._val(iv, "device")
            s = ov.get("samples", "_samples")
            return ["import torch"], [
                f"_z = torch.randn({n}, {ld}).to({dev})",
                f"with torch.no_grad():",
                f"    {s} = {decoder}(_z)",
            ]
        return [], [f"# unknown latent mode {mode!r}"]
