"""Consolidated tensor creation node.

Replaces ones_tensor, zeros_tensor, rand_tensor, and TensorFromListNode.
Pick `fill`:
  zeros | ones | rand   — shape from `shape` (comma-separated ints)
  from_list             — values from `values` (comma-separated floats)
"""
from __future__ import annotations
from core.node import BaseNode, PortType


_FILLS = ["zeros", "ones", "rand", "from_list"]


class TensorCreateNode(BaseNode):
    type_name   = "pt_tensor_create"
    label       = "Create Tensor"
    category    = "Tensor Ops"
    subcategory = ""
    description = (
        "Create a tensor. Pick `fill`:\n"
        "  zeros | ones | rand — shape from `shape` (e.g. '1,64' or '3,32,32')\n"
        "  from_list           — values from `values` (e.g. '0,1,2,3')"
    )

    def relevant_inputs(self, values):
        fill = (values.get("fill") or "zeros").strip().lower()
        if fill == "from_list":
            return ["fill", "values", "requires_grad"]
        return ["fill", "shape", "requires_grad"]

    def _setup_ports(self):
        self.add_input("fill",          PortType.STRING, default="zeros", choices=_FILLS)
        self.add_input("shape",         PortType.STRING, default="1,64", optional=True)
        self.add_input("values",        PortType.STRING, default="0,1,2,3", optional=True)
        self.add_input("requires_grad", PortType.BOOL,   default=False, optional=True)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            fill = (inputs.get("fill") or "zeros").strip().lower()
            rg = bool(inputs.get("requires_grad", False))
            if fill == "from_list":
                vs = inputs.get("values", "0,1,2,3") or "0,1,2,3"
                vals = [float(x.strip()) for x in vs.split(",") if x.strip()]
                t = torch.tensor(vals)
                if rg:
                    t.requires_grad_(True)
                return {"tensor": t}
            shape_str = inputs.get("shape", "1,64") or "1,64"
            shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
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
        fill = (self.inputs["fill"].default_value or "zeros").strip().lower()
        rg = self._val(iv, 'requires_grad')
        if fill == "from_list":
            vs = self._val(iv, 'values')
            return ["import torch"], [
                f"{ov['tensor']} = torch.tensor([float(x) for x in {vs}.split(',') if x.strip()])",
            ]
        shape = self._val(iv, 'shape')
        fn = {"ones": "torch.ones", "rand": "torch.randn"}.get(fill, "torch.zeros")
        return ["import torch"], [
            f"_shape = tuple(int(x) for x in {shape}.split(',') if x.strip())",
            f"{ov['tensor']} = {fn}(_shape, requires_grad={rg})",
        ]
