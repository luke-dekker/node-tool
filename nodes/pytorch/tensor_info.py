"""Tensor Info node — string-format inspection of a tensor."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorInfoNode(BaseNode):
    type_name   = "pt_tensor_info"
    label       = "Tensor Info"
    category    = "Tensor Ops"
    subcategory = ""
    description = (
        "Return tensor info as string. mode='shape' returns just `[1, 2, 3]`; "
        "mode='full' adds dtype + min + max."
    )

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("mode",   PortType.STRING, default="full",
                       choices=["shape", "full"])
        self.add_output("info", PortType.STRING)

    def execute(self, inputs):
        try:
            tensor = inputs.get("tensor")
            mode = (inputs.get("mode") or "full").lower()
            if tensor is None:
                return {"info": "None"}
            if mode == "shape":
                return {"info": str(list(tensor.shape))}
            return {"info":
                    f"shape={list(tensor.shape)} dtype={tensor.dtype} "
                    f"min={tensor.min():.4f} max={tensor.max():.4f}"}
        except Exception:
            return {"info": "error"}

    def export(self, iv, ov):
        t = self._val(iv, 'tensor')
        mode = (self.inputs["mode"].default_value or "full").lower()
        if mode == "shape":
            return [], [f"{ov['info']} = str(list({t}.shape))"]
        return [], [
            f"{ov['info']} = f\"shape={{list({t}.shape)}} dtype={{{t}.dtype}} min={{{t}.min():.4f}} max={{{t}.max():.4f}}\""
        ]
