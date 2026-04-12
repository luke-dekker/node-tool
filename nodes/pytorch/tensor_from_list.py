"""Tensor From List node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorFromListNode(BaseNode):
    type_name   = "pt_tensor_from_list"
    label       = "Tensor From List"
    category    = "Tensor Ops"
    subcategory = ""
    description = "Create a tensor from comma-separated float values"

    def _setup_ports(self):
        self.add_input("values", PortType.STRING, default="0,1,2,3")
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            values_str = inputs.get("values", "0,1,2,3") or "0,1,2,3"
            vals = [float(x.strip()) for x in values_str.split(",") if x.strip()]
            return {"tensor": torch.tensor(vals)}
        except Exception:
            return {"tensor": None}

    def export(self, iv, ov):
        vals = self._val(iv, 'values')
        return ["import torch"], [
            f"{ov['tensor']} = torch.tensor([float(x) for x in {vals}.split(',') if x.strip()])"
        ]
