"""Tensor Info node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorInfoNode(BaseNode):
    type_name   = "pt_tensor_info"
    label       = "Tensor Info"
    category    = "Analyze"
    subcategory = "Tensors"
    description = "Return detailed tensor info as string"

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_output("info", PortType.STRING)

    def execute(self, inputs):
        try:
            tensor = inputs.get("tensor")
            if tensor is None:
                return {"info": "None"}
            info = f"shape={list(tensor.shape)} dtype={tensor.dtype} min={tensor.min():.4f} max={tensor.max():.4f}"
            return {"info": info}
        except Exception:
            return {"info": "error"}

    def export(self, iv, ov):
        t = self._val(iv, 'tensor')
        return [], [
            f"{ov['info']} = f\"shape={{list({t}.shape)}} dtype={{{t}.dtype}} min={{{t}.min():.4f}} max={{{t}.max():.4f}}\""
        ]
