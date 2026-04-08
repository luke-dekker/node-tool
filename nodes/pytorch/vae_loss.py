"""VAE Loss node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class VAELossNode(BaseNode):
    type_name   = "pt_vae_loss"
    label       = "VAE Loss"
    category    = "Models"
    subcategory = "Autoencoder"
    description = "Combined VAE loss: recon_loss + beta * KL. beta=1 is standard VAE, beta>1 is beta-VAE (stronger disentanglement)."

    def _setup_ports(self):
        self.add_input("recon_loss", PortType.TENSOR, default=None)
        self.add_input("kl_loss",    PortType.TENSOR, default=None)
        self.add_input("beta",       PortType.FLOAT,  default=1.0)
        self.add_output("loss", PortType.TENSOR)
        self.add_output("info", PortType.STRING)

    def execute(self, inputs):
        try:
            recon = inputs.get("recon_loss")
            kl    = inputs.get("kl_loss")
            beta  = float(inputs.get("beta") or 1.0)
            if recon is None or kl is None:
                return {"loss": None, "info": "Need recon_loss and kl_loss"}
            loss = recon + beta * kl
            return {"loss": loss, "info": f"loss={loss.item():.6f}  recon={recon.item():.6f}  kl={kl.item():.6f}  beta={beta}"}
        except Exception:
            return {"loss": None, "info": "error"}

    def export(self, iv, ov):
        recon = iv.get("recon_loss") or "None  # TODO: connect recon loss"
        kl    = iv.get("kl_loss")    or "None  # TODO: connect KL loss"
        beta  = self._val(iv, "beta")
        loss  = ov.get("loss", "_vae_loss")
        info  = ov.get("info", "_vae_loss_info")
        return [], [
            f"{loss} = {recon} + {beta} * {kl}",
            f"{info} = f'loss={{{loss}.item():.6f}}  recon={{{recon}.item():.6f}}  "
            f"kl={{{kl}.item():.6f}}  beta={beta}'",
        ]
