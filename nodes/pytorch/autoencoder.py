"""Autoencoder and VAE nodes."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Models"


class AutoencoderNode(BaseNode):
    type_name = "pt_autoencoder"
    label = "Autoencoder"
    category = CATEGORY
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


class VAENode(BaseNode):
    type_name = "pt_vae"
    label = "VAE"
    category = CATEGORY
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


class ReparameterizeNode(BaseNode):
    type_name = "pt_reparameterize"
    label = "Reparameterize"
    category = CATEGORY
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


class KLDivergenceNode(BaseNode):
    type_name = "pt_kl_divergence"
    label = "KL Divergence"
    category = CATEGORY
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


class VAELossNode(BaseNode):
    type_name = "pt_vae_loss"
    label = "VAE Loss"
    category = CATEGORY
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


class VAETrainingConfigNode(BaseNode):
    type_name = "pt_vae_training_config"
    label = "VAE Training Config"
    category = CATEGORY
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


class LatentSamplerNode(BaseNode):
    type_name = "pt_latent_sampler"
    label = "Latent Sampler"
    category = CATEGORY
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


class GaussianNoiseNode(BaseNode):
    type_name = "pt_gaussian_noise"
    label = "Gaussian Noise"
    category = CATEGORY
    description = "Add Gaussian noise to a tensor. Useful for denoising autoencoders: corrupt input, reconstruct clean."

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("std",    PortType.FLOAT,  default=0.1)
        self.add_input("clip",   PortType.BOOL,   default=True)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            t   = inputs.get("tensor")
            std = float(inputs.get("std") or 0.1)
            if t is None:
                return {"tensor": None}
            noisy = t + torch.randn_like(t) * std
            if bool(inputs.get("clip", True)):
                noisy = noisy.clamp(0.0, 1.0)
            return {"tensor": noisy}
        except Exception:
            return {"tensor": None}


# Subcategory stamp
_SC = "Autoencoder"
from core.node import BaseNode as _BN
for _n, _c in list(globals().items()):
    if isinstance(_c, type) and issubclass(_c, _BN) and _c is not _BN:
        _c.subcategory = _SC
