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
