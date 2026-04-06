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
