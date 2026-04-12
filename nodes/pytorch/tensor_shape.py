"""Tensor Shape node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorShapeNode(BaseNode):
    type_name   = "pt_tensor_shape"
    label       = "Tensor Shape"
    category    = "Tensor Ops"
    subcategory = ""
    description = "Return tensor shape as string"

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_output("shape", PortType.STRING)

    def execute(self, inputs):
        try:
            tensor = inputs.get("tensor")
            if tensor is None:
                return {"shape": "None"}
            return {"shape": str(list(tensor.shape))}
        except Exception:
            return {"shape": "error"}

    def export(self, iv, ov):
        return [], [f"{ov['shape']} = str(list({self._val(iv,'tensor')}.shape))"]
