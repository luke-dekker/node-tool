"""Tensor Stack node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorStackNode(BaseNode):
    type_name   = "pt_tensor_stack"
    label       = "Stack"
    category    = "Tensor Ops"
    subcategory = ""
    description = "torch.stack([t1, t2, t3, t4], dim). Stacks tensors along a NEW dimension."

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
            if len(ts) < 2:
                return {"tensor": None}
            return {"tensor": torch.stack(ts, dim=int(inputs.get("dim") or 0))}
        except Exception:
            return {"tensor": None}

    def export(self, iv, ov):
        ts = [iv.get(f"t{i}") for i in range(1, 5) if iv.get(f"t{i}")]
        if len(ts) < 2:
            return ["import torch"], [f"{ov['tensor']} = None  # need >=2 input tensors"]
        return ["import torch"], [
            f"{ov['tensor']} = torch.stack([{', '.join(ts)}], dim={self._val(iv, 'dim')})"
        ]
