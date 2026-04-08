"""Freeze Backbone node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class FreezeBackboneNode(BaseNode):
    type_name   = "pt_freeze_backbone"
    label       = "Freeze Layers"
    category    = "Models"
    subcategory = "Pretrained"
    description = "Freeze all parameters in a model (requires_grad=False). Pass through to unfreeze_from node or directly to optimizer."

    def _setup_ports(self):
        self.add_input("model",        PortType.MODULE, default=None)
        self.add_input("freeze_all",   PortType.BOOL,   default=True)
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING)

    def execute(self, inputs):
        try:
            model = inputs.get("model")
            if model is None:
                return {"model": None, "info": "No model"}
            freeze = bool(inputs.get("freeze_all", True))
            for p in model.parameters():
                p.requires_grad = not freeze
            frozen = sum(1 for p in model.parameters() if not p.requires_grad)
            total  = sum(1 for p in model.parameters())
            return {"model": model, "info": f"frozen={frozen}/{total} param groups"}
        except Exception:
            return {"model": None, "info": "error"}

    def export(self, iv, ov):
        in_model = iv.get("model") or "None  # TODO: connect a model"
        mv = ov.get("model", "_frozen_model")
        iv_var = ov.get("info",  "_frozen_model_info")
        freeze = bool(self.inputs["freeze_all"].default_value)
        lines = [
            f"{mv} = {in_model}",
            f"for _p in {mv}.parameters():",
            f"    _p.requires_grad = {not freeze}",
            f"_n_frozen = sum(1 for _p in {mv}.parameters() if not _p.requires_grad)",
            f"_n_total  = sum(1 for _p in {mv}.parameters())",
            f"{iv_var} = f'frozen={{_n_frozen}}/{{_n_total}} param groups'",
        ]
        return ["import torch"], lines
