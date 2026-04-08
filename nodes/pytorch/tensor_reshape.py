"""Tensor Reshape node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorReshapeNode(BaseNode):
    type_name   = "pt_tensor_reshape"
    label       = "Reshape"
    category    = "Analyze"
    subcategory = "Tensors"
    description = "tensor.reshape(shape). Enter shape as comma-separated ints, e.g. '32,-1' or '2,3,4'. Use -1 for inferred dim."

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("shape",  PortType.STRING, default="-1")
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            t = inputs.get("tensor")
            if t is None:
                return {"tensor": None}
            shape = [int(x.strip()) for x in str(inputs.get("shape") or "-1").split(",")]
            return {"tensor": t.reshape(shape)}
        except Exception:
            return {"tensor": None}

    def export(self, iv, ov):
        t = iv.get("tensor") or "None"
        shape_str = str(self.inputs["shape"].default_value or "-1")
        shape_tuple = ", ".join(s.strip() for s in shape_str.split(","))
        return ["import torch"], [
            f"{ov['tensor']} = {t}.reshape({shape_tuple})"
        ]
