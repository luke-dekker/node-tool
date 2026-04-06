"""Freeze Named Layers node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class FreezeNamedLayersNode(BaseNode):
    type_name   = "pt_freeze_named_layers"
    label       = "Freeze Named Layers"
    category    = "Models"
    subcategory = "Pretrained"
    description = "Freeze specific layers by name prefix. Enter comma-separated prefixes, e.g. 'encoder,layer1'. Unmatched layers stay unchanged."

    def _setup_ports(self):
        self.add_input("model",   PortType.MODULE, default=None)
        self.add_input("names",   PortType.STRING, default="encoder")
        self.add_input("freeze",  PortType.BOOL,   default=True)
        self.add_output("model",  PortType.MODULE)
        self.add_output("info",   PortType.STRING)

    def execute(self, inputs):
        try:
            model = inputs.get("model")
            if model is None:
                return {"model": None, "info": "No model"}
            names = [n.strip() for n in str(inputs.get("names") or "").split(",") if n.strip()]
            freeze = bool(inputs.get("freeze", True))
            count = 0
            for name, param in model.named_parameters():
                if any(name.startswith(prefix) for prefix in names):
                    param.requires_grad = not freeze
                    count += 1
            action = "Frozen" if freeze else "Unfrozen"
            return {"model": model, "info": f"{action} {count} params matching {names}"}
        except Exception:
            return {"model": None, "info": "error"}
