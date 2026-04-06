"""VAE node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class VAENode(BaseNode):
    type_name   = "pt_vae"
    label       = "VAE"
    category    = "Models"
    subcategory = "Autoencoder"
    description = ("Builds a Variational Autoencoder. layer_sizes defines the encoder depth. "
                   "E.g. '784,256,64' with latent_dim=16 creates encoder 784->256->64->(mu,log_var) "
                   "and mirror decoder. Forward returns (reconstruction, mu, log_var).")

    def _setup_ports(self):
        self.add_input("layer_sizes", PortType.STRING, default="784,256,64")
        self.add_input("latent_dim",  PortType.INT,    default=16)
        self.add_input("activation",  PortType.STRING, default="relu")
        self.add_output("model",      PortType.MODULE)
        self.add_output("encoder",    PortType.MODULE)
        self.add_output("decoder",    PortType.MODULE)
        self.add_output("info",       PortType.STRING)

    def execute(self, inputs):
        try:
            import torch
            import torch.nn as nn

            sizes = [int(x.strip()) for x in str(inputs.get("layer_sizes") or "784,256,64").split(",")]
            latent_dim = int(inputs.get("latent_dim") or 16)
            act_name = str(inputs.get("activation") or "relu").lower()
            act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "leaky_relu": nn.LeakyReLU}
            Act = act_map.get(act_name, nn.ReLU)

            # Encoder backbone (no final activation)
            enc_layers = []
            for i in range(len(sizes) - 1):
                enc_layers.append(nn.Linear(sizes[i], sizes[i+1]))
                enc_layers.append(Act())
            encoder_backbone = nn.Sequential(*enc_layers)
            enc_out_dim = sizes[-1]

            # mu and log_var projection heads
            mu_layer      = nn.Linear(enc_out_dim, latent_dim)
            log_var_layer = nn.Linear(enc_out_dim, latent_dim)

            # Decoder (mirror of sizes + latent_dim at input)
            dec_sizes = [latent_dim] + list(reversed(sizes))
            dec_layers = []
            for i in range(len(dec_sizes) - 1):
                dec_layers.append(nn.Linear(dec_sizes[i], dec_sizes[i+1]))
                if i < len(dec_sizes) - 2:
                    dec_layers.append(Act())
            decoder = nn.Sequential(*dec_layers)

            class VAE(nn.Module):
                def __init__(self, backbone, mu_l, lv_l, dec):
                    super().__init__()
                    self.encoder   = backbone
                    self.mu_layer  = mu_l
                    self.lv_layer  = lv_l
                    self.decoder   = dec

                def reparameterize(self, mu, log_var):
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn_like(std)
                    return mu + eps * std

                def forward(self, x):
                    h       = self.encoder(x)
                    mu      = self.mu_layer(h)
                    log_var = self.lv_layer(h)
                    z       = self.reparameterize(mu, log_var)
                    recon   = self.decoder(z)
                    return recon, mu, log_var

                def encode(self, x):
                    h = self.encoder(x)
                    return self.mu_layer(h), self.lv_layer(h)

                def decode(self, z):
                    return self.decoder(z)

                def sample(self, n, device="cpu"):
                    z = torch.randn(n, self.mu_layer.out_features).to(device)
                    return self.decoder(z)

            model = VAE(encoder_backbone, mu_layer, log_var_layer, decoder)
            params = sum(p.numel() for p in model.parameters())
            info = f"VAE {sizes}->[{latent_dim}]  params={params:,}"
            return {"model": model, "encoder": encoder_backbone, "decoder": decoder, "info": info}
        except Exception:
            import traceback
            return {"model": None, "encoder": None, "decoder": None,
                    "info": traceback.format_exc().split("\n")[-2]}
