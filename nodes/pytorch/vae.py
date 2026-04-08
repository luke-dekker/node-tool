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

    def export(self, iv, ov):
        # NOTE: like AutoencoderNode, this packs an entire VAE into a single nn.Module.
        # Faithful export emits the same internal class so behavior matches runtime exactly.
        sizes_str = str(self.inputs["layer_sizes"].default_value or "784,256,64")
        latent_dim = self._val(iv, "latent_dim")
        act_name = str(self.inputs["activation"].default_value or "relu").lower()
        act_cls = {"relu": "nn.ReLU", "tanh": "nn.Tanh",
                   "sigmoid": "nn.Sigmoid", "leaky_relu": "nn.LeakyReLU"}.get(act_name, "nn.ReLU")
        cls_name = f"_VAE_{self.id[:6]}"
        m_var = ov.get("model",   "_vae")
        e_var = ov.get("encoder", "_vae_encoder")
        d_var = ov.get("decoder", "_vae_decoder")
        i_var = ov.get("info",    "_vae_info")

        return ["import torch", "import torch.nn as nn"], [
            f"_sizes = [{sizes_str}]",
            f"_enc_layers = []",
            f"for _i in range(len(_sizes) - 1):",
            f"    _enc_layers.append(nn.Linear(_sizes[_i], _sizes[_i+1]))",
            f"    _enc_layers.append({act_cls}())",
            f"{e_var} = nn.Sequential(*_enc_layers)",
            f"_mu_layer  = nn.Linear(_sizes[-1], {latent_dim})",
            f"_lv_layer  = nn.Linear(_sizes[-1], {latent_dim})",
            f"_dec_sizes = [{latent_dim}] + list(reversed(_sizes))",
            f"_dec_layers = []",
            f"for _i in range(len(_dec_sizes) - 1):",
            f"    _dec_layers.append(nn.Linear(_dec_sizes[_i], _dec_sizes[_i+1]))",
            f"    if _i < len(_dec_sizes) - 2:",
            f"        _dec_layers.append({act_cls}())",
            f"{d_var} = nn.Sequential(*_dec_layers)",
            f"",
            f"class {cls_name}(nn.Module):",
            f"    def __init__(self, backbone, mu_l, lv_l, dec):",
            f"        super().__init__()",
            f"        self.encoder  = backbone",
            f"        self.mu_layer = mu_l",
            f"        self.lv_layer = lv_l",
            f"        self.decoder  = dec",
            f"    def reparameterize(self, mu, log_var):",
            f"        std = torch.exp(0.5 * log_var)",
            f"        eps = torch.randn_like(std)",
            f"        return mu + eps * std",
            f"    def forward(self, x):",
            f"        h       = self.encoder(x)",
            f"        mu      = self.mu_layer(h)",
            f"        log_var = self.lv_layer(h)",
            f"        z       = self.reparameterize(mu, log_var)",
            f"        return self.decoder(z), mu, log_var",
            f"",
            f"{m_var} = {cls_name}({e_var}, _mu_layer, _lv_layer, {d_var})",
            f"{i_var} = f'VAE {{_sizes}}->[{latent_dim}]  "
            f"params={{sum(p.numel() for p in {m_var}.parameters()):,}}'",
        ]
