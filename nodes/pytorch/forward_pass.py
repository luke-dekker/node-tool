"""Forward Pass node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ForwardPassNode(BaseNode):
    type_name   = "pt_forward_pass"
    label       = "Forward Pass"
    category    = "Training"
    subcategory = "Config"
    description = "Run model.eval() forward pass on input tensor"

    def _setup_ports(self):
        self.add_input("model", PortType.MODULE, default=None)
        self.add_input("input", PortType.TENSOR, default=None)
        self.add_output("output", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            model = inputs.get("model")
            inp = inputs.get("input")
            if model is None or inp is None:
                return {"output": None}
            model.eval()
            with torch.no_grad():
                out = model(inp)
            return {"output": out}
        except Exception:
            return {"output": None}

    def export(self, iv, ov):
        m = self._val(iv, 'model')
        inp = self._val(iv, 'input')
        return ["import torch"], [
            f"{m}.eval()",
            f"with torch.no_grad():",
            f"    {ov['output']} = {m}({inp})",
        ]
