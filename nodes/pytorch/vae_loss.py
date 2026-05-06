"""VAE-related loss node — combines KLDivergenceNode and VAELossNode.

Pick `mode`:
  kl       — compute -0.5 * mean(1 + log_variance - mean^2 - exp(log_variance))
             from `mean` + `log_variance` → kl_loss
  combine  — recon_loss + beta * kl_loss → total VAE loss
"""
from __future__ import annotations
from core.node import BaseNode, PortType


_MODES = ["kl", "combine"]


class VAELossNode(BaseNode):
    type_name   = "pt_vae_loss"
    label       = "VAE Loss"
    category    = "Models"
    subcategory = "Autoencoder"
    description = (
        "Pick `mode`:\n"
        "  kl       — compute KL divergence from encoder's mean / log_variance\n"
        "  combine  — total VAE loss = recon_loss + beta * kl_loss"
    )

    def relevant_inputs(self, values):
        mode = (values.get("mode") or "combine").strip()
        if mode == "kl":      return ["mode"]            # mean + log_variance wired
        if mode == "combine": return ["mode", "beta"]    # recon_loss + kl_loss wired
        return ["mode"]

    def _setup_ports(self):
        self.add_input("mode",         PortType.STRING, "combine", choices=_MODES)
        # kl mode
        self.add_input("mean",         PortType.TENSOR, default=None, optional=True)
        self.add_input("log_variance", PortType.TENSOR, default=None, optional=True)
        # combine mode
        self.add_input("recon_loss",   PortType.TENSOR, default=None, optional=True)
        self.add_input("kl_loss",      PortType.TENSOR, default=None, optional=True)
        self.add_input("beta",         PortType.FLOAT,  1.0,          optional=True)
        self.add_output("loss",    PortType.TENSOR,
                        description="combine: total VAE loss; kl: identical to kl_loss")
        self.add_output("kl_loss", PortType.TENSOR,
                        description="kl mode: computed KL term; combine: passthrough")
        self.add_output("info",    PortType.STRING)

    def execute(self, inputs):
        out = {"loss": None, "kl_loss": None, "info": ""}
        try:
            import torch
            mode = (inputs.get("mode") or "combine").strip()
            if mode == "kl":
                mu     = inputs.get("mean")
                logvar = inputs.get("log_variance")
                if mu is None or logvar is None:
                    out["info"] = "Need mean and log_variance"; return out
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                out["kl_loss"] = kl
                out["loss"]    = kl       # convenience: also expose on `loss`
                out["info"]    = f"KL={kl.item():.6f}"
                return out
            if mode == "combine":
                recon = inputs.get("recon_loss")
                kl    = inputs.get("kl_loss")
                beta  = float(inputs.get("beta") or 1.0)
                if recon is None or kl is None:
                    out["info"] = "Need recon_loss and kl_loss"; return out
                loss = recon + beta * kl
                out["loss"]    = loss
                out["kl_loss"] = kl       # passthrough
                out["info"]    = (f"loss={loss.item():.6f} recon={recon.item():.6f} "
                                  f"kl={kl.item():.6f} beta={beta}")
                return out
            return out
        except Exception:
            return {"loss": None, "kl_loss": None, "info": "error"}

    def export(self, iv, ov):
        mode = (self.inputs["mode"].default_value or "combine")
        loss = ov.get("loss", "_vae_loss"); kl = ov.get("kl_loss", "_kl"); info = ov.get("info", "_vae_info")
        if mode == "kl":
            mu = iv.get("mean") or "None"; lv = iv.get("log_variance") or "None"
            return ["import torch"], [
                f"{kl} = -0.5 * torch.mean(1 + {lv} - {mu}.pow(2) - {lv}.exp())",
                f"{loss} = {kl}",
                f"{info} = f'KL={{{kl}.item():.6f}}'",
            ]
        if mode == "combine":
            recon = iv.get("recon_loss") or "None"; kli = iv.get("kl_loss") or "None"
            beta = self._val(iv, "beta")
            return [], [
                f"{loss} = {recon} + {beta} * {kli}",
                f"{kl}   = {kli}",
                f"{info} = f'loss={{{loss}.item():.6f}} recon={{{recon}.item():.6f}} "
                f"kl={{{kli}.item():.6f}} beta={beta}'",
            ]
        return [], [f"# unknown vae_loss mode {mode!r}"]
