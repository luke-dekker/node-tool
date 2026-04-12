"""Tensor Concatenate node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorCatNode(BaseNode):
    type_name   = "pt_tensor_cat"
    label       = "Concatenate"
    category    = "Tensor Ops"
    subcategory = ""
    description = "torch.cat([t1, t2, t3, t4], dim). Concatenates up to 4 tensors along dim."

    def _setup_ports(self):
        self.add_input("t1",  PortType.TENSOR, default=None)
        self.add_input("t2",  PortType.TENSOR, default=None)
        self.add_input("t3",  PortType.TENSOR, default=None)
        self.add_input("t4",  PortType.TENSOR, default=None)
        self.add_input("dim", PortType.INT,    default=0)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            ts = [inputs.get(f"t{i}") for i in range(1, 5) if inputs.get(f"t{i}") is not None]
            if not ts:
                return {"tensor": None}
            return {"tensor": torch.cat(ts, dim=int(inputs.get("dim") or 0))}
        except Exception:
            return {"tensor": None}

    def export(self, iv, ov):
        # Only emit the tensors actually connected upstream
        ts = [iv.get(f"t{i}") for i in range(1, 5) if iv.get(f"t{i}")]
        if not ts:
            return ["import torch"], [f"{ov['tensor']} = None  # no input tensors connected"]
        return ["import torch"], [
            f"{ov['tensor']} = torch.cat([{', '.join(ts)}], dim={self._val(iv, 'dim')})"
        ]
