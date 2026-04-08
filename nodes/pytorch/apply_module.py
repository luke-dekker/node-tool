"""ApplyModuleNode — bridges a MODULE-typed node into a tensor flow chain.

The gap this fills: nodes like ResNet18Node, MobileNetV3Node, ClassModuleImportNode
output a `model: MODULE` but have no tensor_in/tensor_out. They can't sit in a
forward chain on their own. Wire one of those into ApplyModule's `model` input,
wire your tensor into `tensor_in`, and the node calls `model(tensor_in)` on every
forward pass — exactly what GraphAsModule needs to run a backbone end-to-end.

Parameter discovery happens through the backbone's own _layer attribute, so this
node deliberately does NOT register the model as its own _layer (that would
double-count parameters in the optimizer).
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class ApplyModuleNode(BaseNode):
    type_name   = "pt_apply_module"
    label       = "Apply Module"
    category    = "Layers"
    subcategory = "Bridges"
    description = (
        "Apply a wired MODULE to a tensor — bridges backbone/model nodes into "
        "the tensor_in/tensor_out flow expected by training graphs."
    )

    def _setup_ports(self) -> None:
        self.add_input("model",     PortType.MODULE, default=None,
                       description="Wire from a backbone, ClassModuleImport, etc.")
        self.add_input("tensor_in", PortType.TENSOR, default=None)
        self.add_output("tensor_out",  PortType.TENSOR)
        self.add_output("input_shape",  PortType.STRING)
        self.add_output("output_shape", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        model = inputs.get("model")
        tensor = inputs.get("tensor_in")
        empty = {"tensor_out": None, "input_shape": "", "output_shape": ""}
        if model is None or tensor is None:
            return empty
        try:
            out = model(tensor)
            return {
                "tensor_out":   out,
                "input_shape":  f"({', '.join(str(d) for d in tensor.shape)})",
                "output_shape": f"({', '.join(str(d) for d in out.shape)})",
            }
        except Exception as exc:
            return {**empty, "input_shape": f"forward failed: {exc}"}

    def export(self, iv, ov):
        model = iv.get("model")     or "None  # TODO: connect a model"
        tin   = iv.get("tensor_in") or "None  # TODO: connect input tensor"
        out_var      = ov.get("tensor_out",   "_apply_out")
        in_shape_var = ov.get("input_shape",  "_apply_in_shape")
        out_shape_var= ov.get("output_shape", "_apply_out_shape")
        return ["import torch"], [
            f"{out_var}       = {model}({tin})",
            f"{in_shape_var}  = f'({{\", \".join(str(d) for d in {tin}.shape)}})'",
            f"{out_shape_var} = f'({{\", \".join(str(d) for d in {out_var}.shape)}})'",
        ]
