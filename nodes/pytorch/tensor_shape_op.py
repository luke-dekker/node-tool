"""Consolidated tensor shape operation — replaces tensor_squeeze, tensor_unsqueeze,
tensor_reshape."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorShapeOpNode(BaseNode):
    type_name   = "pt_tensor_shape_op"
    label       = "Shape Op"
    category    = "Tensor Ops"
    subcategory = ""
    description = (
        "Reshape, squeeze, or unsqueeze a tensor. "
        "mode='reshape': uses shape string (comma-separated, -1 for inferred). "
        "mode='squeeze': removes size-1 dim at dim (-1 = all). "
        "mode='unsqueeze': inserts size-1 dim at dim."
    )

    def _setup_ports(self):
        self.add_input("mode",   PortType.STRING, default="reshape")
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("dim",    PortType.INT,    default=0)
        self.add_input("shape",  PortType.STRING, default="-1")
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            t = inputs.get("tensor")
            if t is None:
                return {"tensor": None}
            mode = (inputs.get("mode") or "reshape").strip().lower()
            if mode == "squeeze":
                dim = int(inputs.get("dim") if inputs.get("dim") is not None else -1)
                return {"tensor": t.squeeze(dim) if dim >= 0 else t.squeeze()}
            elif mode == "unsqueeze":
                dim = int(inputs.get("dim") or 0)
                return {"tensor": t.unsqueeze(dim)}
            else:  # reshape
                shape = [int(x.strip()) for x in str(inputs.get("shape") or "-1").split(",")]
                return {"tensor": t.reshape(shape)}
        except Exception:
            return {"tensor": None}

    def export(self, iv, ov):
        t = iv.get("tensor") or "None"
        mode = (self.inputs["mode"].default_value or "reshape").strip().lower()
        if mode == "squeeze":
            dim = self.inputs["dim"].default_value
            if dim is None or int(dim) < 0:
                return ["import torch"], [f"{ov['tensor']} = {t}.squeeze()"]
            return ["import torch"], [f"{ov['tensor']} = {t}.squeeze({dim})"]
        elif mode == "unsqueeze":
            return ["import torch"], [
                f"{ov['tensor']} = {t}.unsqueeze({self._val(iv, 'dim')})"
            ]
        else:
            shape_str = str(self.inputs["shape"].default_value or "-1")
            shape_tuple = ", ".join(s.strip() for s in shape_str.split(","))
            return ["import torch"], [f"{ov['tensor']} = {t}.reshape({shape_tuple})"]
