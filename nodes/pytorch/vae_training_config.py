"""VAE Training Config node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class VAETrainingConfigNode(BaseNode):
    type_name   = "pt_vae_training_config"
    label       = "VAE Training Config"
    category    = "Models"
    subcategory = "Autoencoder"
    description = ("Training config for VAE models. Model must return (reconstruction, mu, log_var). "
                   "Loss = recon_loss(recon, x) + beta * KL(mu, log_var). "
                   "recon_loss_type: 'mse' or 'bce'.")

    def _setup_ports(self):
        self.add_input("model",           PortType.MODULE,     default=None)
        self.add_input("optimizer",       PortType.OPTIMIZER,  default=None)
        self.add_input("dataloader",      PortType.DATALOADER, default=None)
        self.add_input("val_dataloader",  PortType.DATALOADER, default=None)
        self.add_input("scheduler",       PortType.SCHEDULER,  default=None)
        self.add_input("epochs",          PortType.INT,        default=20)
        self.add_input("beta",            PortType.FLOAT,      default=1.0)
        self.add_input("recon_loss_type", PortType.STRING,     default="mse")
        self.add_input("device",          PortType.STRING,     default="cpu")
        self.add_output("config", PortType.ANY)

    def execute(self, inputs):
        try:
            return {"config": {
                "vae": True,
                "model":           inputs.get("model"),
                "optimizer":       inputs.get("optimizer"),
                "dataloader":      inputs.get("dataloader"),
                "val_dataloader":  inputs.get("val_dataloader"),
                "scheduler":       inputs.get("scheduler"),
                "epochs":          int(inputs.get("epochs") or 20),
                "beta":            float(inputs.get("beta") or 1.0),
                "recon_loss_type": str(inputs.get("recon_loss_type") or "mse"),
                "device":          str(inputs.get("device") or "cpu"),
            }}
        except Exception:
            return {"config": None}
