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

    def export(self, iv, ov):
        # NOTE: this node packs an entire encoder+decoder into a single nn.Module.
        # That's the one place in the codebase where graph-IS-the-model is violated.
        # Faithful export emits the same internal Sequentials so behavior matches the
        # runtime exactly. A future refactor could split this into per-layer nodes.
        sizes_str = str(self.inputs["layer_sizes"].default_value or "784,256,64,16")
        act_name = str(self.inputs["activation"].default_value or "relu").lower()
        act_cls = {"relu": "nn.ReLU", "tanh": "nn.Tanh",
                   "sigmoid": "nn.Sigmoid", "leaky_relu": "nn.LeakyReLU"}.get(act_name, "nn.ReLU")
        m_var = ov.get("model",   "_ae")
        e_var = ov.get("encoder", "_ae_encoder")
        d_var = ov.get("decoder", "_ae_decoder")
        l_var = ov.get("latent_dim", "_ae_latent_dim")
        i_var = ov.get("info",    "_ae_info")

        return ["import torch", "import torch.nn as nn"], [
            f"_sizes = [{sizes_str}]",
            f"_enc_layers = []",
            f"for _i in range(len(_sizes) - 1):",
            f"    _enc_layers.append(nn.Linear(_sizes[_i], _sizes[_i+1]))",
            f"    if _i < len(_sizes) - 2:",
            f"        _enc_layers.append({act_cls}())",
            f"{e_var} = nn.Sequential(*_enc_layers)",
            f"_rev = list(reversed(_sizes))",
            f"_dec_layers = []",
            f"for _i in range(len(_rev) - 1):",
            f"    _dec_layers.append(nn.Linear(_rev[_i], _rev[_i+1]))",
            f"    if _i < len(_rev) - 2:",
            f"        _dec_layers.append({act_cls}())",
            f"{d_var} = nn.Sequential(*_dec_layers)",
            f"",
            f"class _AE_{self.id[:6]}(nn.Module):",
            f"    def __init__(self, enc, dec):",
            f"        super().__init__()",
            f"        self.encoder = enc",
            f"        self.decoder = dec",
            f"    def forward(self, x):",
            f"        return self.decoder(self.encoder(x))",
            f"",
            f"{m_var} = _AE_{self.id[:6]}({e_var}, {d_var})",
            f"{l_var} = _sizes[-1]",
            f"{i_var} = f'AE {{_sizes}}  latent={{_sizes[-1]}}  "
            f"params={{sum(p.numel() for p in {m_var}.parameters()):,}}'",
        ]
