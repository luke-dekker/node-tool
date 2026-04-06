"""Pretrained Block node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class PretrainedBlockNode(BaseNode):
    """Load a full saved model as a drop-in block with freeze controls."""
    type_name   = "pt_pretrained_block"
    label       = "Pretrained Block"
    category    = "Models"
    subcategory = "Pretrained"
    description = (
        "Load a model saved with Save Full Model. "
        "Freeze all or only the first N layers for transfer learning / fine-tuning."
    )

    def _setup_ports(self) -> None:
        self.add_input("path",             PortType.STRING, default="model_full.pt",
                       description="Path to a .pt file saved with Save Full Model")
        self.add_input("device",           PortType.STRING, default="cpu")
        self.add_input("freeze_all",       PortType.BOOL,   default=False,
                       description="Freeze every parameter (feature extractor mode)")
        self.add_input("trainable_layers", PortType.INT,    default=0,
                       description="Unfreeze this many layers from the end (0 = respect freeze_all)")
        self.add_input("eval_mode",        PortType.BOOL,   default=False,
                       description="Force model.eval() — disables dropout / batchnorm training noise")
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING,
                        description="Class, param count, frozen/trainable summary")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        path             = inputs.get("path") or "model_full.pt"
        device           = inputs.get("device") or "cpu"
        freeze_all       = bool(inputs.get("freeze_all", False))
        trainable_layers = int(inputs.get("trainable_layers") or 0)
        eval_mode        = bool(inputs.get("eval_mode", False))

        try:
            model = torch.load(path, map_location=device, weights_only=False)
            model.to(device)
        except Exception as exc:
            return {"model": None, "info": f"Load failed: {exc}"}

        # Apply freezing
        if freeze_all:
            for p in model.parameters():
                p.requires_grad = False

        # Unfreeze the last N layers (by named children, then parameters)
        if trainable_layers > 0:
            children = list(model.children())
            for child in children[-trainable_layers:]:
                for p in child.parameters():
                    p.requires_grad = True

        if eval_mode:
            model.eval()

        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen    = total - trainable
        mode_str  = "eval" if eval_mode else "train"
        info = (
            f"{model.__class__.__name__} loaded from {path}\n"
            f"Device: {device} | Mode: {mode_str}\n"
            f"Total:     {total:,} params\n"
            f"Trainable: {trainable:,} params\n"
            f"Frozen:    {frozen:,} params"
        )
        return {"model": model, "info": info}

    def export(self, iv, ov):
        path             = self._val(iv, "path")
        device           = self._val(iv, "device")
        freeze_all       = self._val(iv, "freeze_all")
        trainable_layers = self._val(iv, "trainable_layers")
        eval_mode        = self._val(iv, "eval_mode")
        out_m = ov.get("model", "_pretrained")
        lines = [
            f"{out_m} = torch.load({path}, map_location={device}, weights_only=False)",
            f"{out_m}.to({device})",
            f"if {freeze_all}:",
            f"    for _p in {out_m}.parameters(): _p.requires_grad = False",
            f"if {trainable_layers} > 0:",
            f"    for _child in list({out_m}.children())[-{trainable_layers}:]:",
            f"        for _p in _child.parameters(): _p.requires_grad = True",
            f"if {eval_mode}: {out_m}.eval()",
        ]
        return ["import torch"], lines
