"""Autoencoder node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class AutoencoderNode(BaseNode):
    type_name   = "pt_autoencoder"
    label       = "Autoencoder"
    category    = "Models"
    subcategory = "Autoencoder"
    description = ("Builds a symmetric MLP autoencoder from a comma-separated list of layer sizes. "
                   "E.g. '784,256,64,16' creates encoder 784->256->64->16 and "
                   "decoder 16->64->256->784. Uses ReLU between layers, no activation on output.")

    def _setup_ports(self):
        self.add_input("layer_sizes", PortType.STRING, default="784,256,64,16")
        self.add_input("activation",  PortType.STRING, default="relu")
        self.add_output("model",        PortType.MODULE)
        self.add_output("encoder",      PortType.MODULE)
        self.add_output("decoder",      PortType.MODULE)
        self.add_output("latent_dim",   PortType.INT)
        self.add_output("info",         PortType.STRING)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            sizes = [int(x.strip()) for x in str(inputs.get("layer_sizes") or "784,256,64").split(",")]
            act_name = str(inputs.get("activation") or "relu").lower()
            act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "leaky_relu": nn.LeakyReLU}
            Act = act_map.get(act_name, nn.ReLU)

            # Build encoder
            enc_layers = []
            for i in range(len(sizes) - 1):
                enc_layers.append(nn.Linear(sizes[i], sizes[i+1]))
                if i < len(sizes) - 2:
                    enc_layers.append(Act())
            encoder = nn.Sequential(*enc_layers)

            # Build decoder (mirror)
            rev = list(reversed(sizes))
            dec_layers = []
            for i in range(len(rev) - 1):
                dec_layers.append(nn.Linear(rev[i], rev[i+1]))
                if i < len(rev) - 2:
                    dec_layers.append(Act())
            decoder = nn.Sequential(*dec_layers)

            class AE(nn.Module):
                def __init__(self, enc, dec):
                    super().__init__()
                    self.encoder = enc
                    self.decoder = dec
                def forward(self, x):
                    return self.decoder(self.encoder(x))

            model = AE(encoder, decoder)
            params = sum(p.numel() for p in model.parameters())
            info = f"AE {sizes}  latent={sizes[-1]}  params={params:,}"
            return {"model": model, "encoder": encoder, "decoder": decoder,
                    "latent_dim": sizes[-1], "info": info}
        except Exception:
            import traceback
            return {"model": None, "encoder": None, "decoder": None,
                    "latent_dim": 0, "info": traceback.format_exc().split("\n")[-2]}
