"""Consolidated tensor creation — replaces ones_tensor, zeros_tensor, rand_tensor."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorCreateNode(BaseNode):
    type_name   = "pt_tensor_create"
    label       = "Create Tensor"
    category    = "Tensor Ops"
    subcategory = ""
    description = (
        "Create a tensor filled with zeros, ones, or random values. "
        "Shape as comma-separated ints, e.g. '1,64' or '3,32,32'."
    )

    def _setup_ports(self):
        self.add_input("fill",          PortType.STRING, default="zeros")
        self.add_input("shape",         PortType.STRING, default="1,64")
        self.add_input("requires_grad", PortType.BOOL,   default=False)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            shape_str = inputs.get("shape", "1,64") or "1,64"
            shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
            fill = (inputs.get("fill") or "zeros").strip().lower()
            rg = bool(inputs.get("requires_grad", False))
            if fill == "ones":
                t = torch.ones(shape)
            elif fill == "rand":
                t = torch.randn(shape, requires_grad=rg)
            else:
                t = torch.zeros(shape)
            if fill != "rand" and rg:
                t.requires_grad_(True)
            return {"tensor": t}
        except Exception:
            return {"tensor": None}

    def export(self, iv, ov):
        shape = self._val(iv, 'shape')
        fill = (self.inputs["fill"].default_value or "zeros").strip().lower()
        rg = self._val(iv, 'requires_grad')
        fn = {"ones": "torch.ones", "rand": "torch.randn"}.get(fill, "torch.zeros")
        return ["import torch"], [
            f"_shape = tuple(int(x) for x in {shape}.split(',') if x.strip())",
            f"{ov['tensor']} = {fn}(_shape, requires_grad={rg})",
        ]
