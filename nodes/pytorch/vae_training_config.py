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

    def export(self, iv, ov):
        # Emit a real VAE training loop. Mirrors the runtime behavior of the GUI's
        # training panel for VAE configs: forward returns (recon, mu, log_var),
        # loss = recon_loss + beta * KL.
        model = iv.get("model")     or "None  # TODO: connect a VAE model"
        opt   = iv.get("optimizer") or "None  # TODO: connect an optimizer"
        dl    = iv.get("dataloader") or "None  # TODO: connect a dataloader"
        val_dl = iv.get("val_dataloader")
        sched  = iv.get("scheduler")
        epochs = self._val(iv, "epochs")
        beta   = self._val(iv, "beta")
        recon_type = str(self.inputs["recon_loss_type"].default_value or "mse").lower()
        device = self._val(iv, "device")
        cfg_var = ov.get("config", "_vae_config")

        recon_call = ("F.binary_cross_entropy(_recon, _x.view(_recon.shape), reduction='mean')"
                      if recon_type == "bce"
                      else "F.mse_loss(_recon, _x.view(_recon.shape))")

        lines = [
            f"# VAE training config",
            f"{cfg_var} = {{'vae': True, 'epochs': {epochs}, 'beta': {beta}}}",
            f"_model = {model}.to({device})",
            f"_opt   = {opt}",
            f"for _epoch in range({epochs}):",
            f"    _model.train()",
            f"    _running = 0.0",
            f"    for _batch in {dl}:",
            f"        _x = _batch[0] if isinstance(_batch, (list, tuple)) else _batch",
            f"        _x = _x.to({device})",
            f"        _opt.zero_grad()",
            f"        _recon, _mu, _lv = _model(_x)",
            f"        _recon_loss = {recon_call}",
            f"        _kl = -0.5 * torch.mean(1 + _lv - _mu.pow(2) - _lv.exp())",
            f"        _loss = _recon_loss + {beta} * _kl",
            f"        _loss.backward()",
            f"        _opt.step()",
            f"        _running += _loss.item()",
            f"    print(f'epoch {{_epoch+1}}/{epochs}  loss={{_running / max(1, len({dl})):.6f}}')",
        ]
        if val_dl:
            lines += [
                f"    _model.eval()",
                f"    with torch.no_grad():",
                f"        _val_loss = 0.0",
                f"        for _batch in {val_dl}:",
                f"            _x = _batch[0] if isinstance(_batch, (list, tuple)) else _batch",
                f"            _x = _x.to({device})",
                f"            _r, _mu, _lv = _model(_x)",
                f"            _val_loss += ({recon_call}).item()",
                f"        print(f'  val={{_val_loss / max(1, len({val_dl})):.6f}}')",
            ]
        if sched:
            lines.append(f"    {sched}.step()")
        return ["import torch", "import torch.nn.functional as F"], lines
