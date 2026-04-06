"""Export ONNX node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class ExportONNXNode(BaseNode):
    """Export a model to ONNX format for deployment."""
    type_name   = "pt_export_onnx"
    label       = "Export ONNX"
    category    = "Models"
    subcategory = "Save & Load"
    description = (
        "Export model to ONNX for deployment on hardware / inference engines. "
        "Provide a dummy input matching your model's expected input shape."
    )

    def _setup_ports(self) -> None:
        self.add_input("model",       PortType.MODULE, default=None)
        self.add_input("input_shape", PortType.STRING, default="1,784",
                       description="Dummy input shape, e.g. 1,3,224,224")
        self.add_input("path",        PortType.STRING, default="model.onnx")
        self.add_input("opset",       PortType.INT,    default=17,
                       description="ONNX opset version")
        self.add_output("path",       PortType.STRING, description="Path exported to")
        self.add_output("info",       PortType.STRING, description="Export summary")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        model     = inputs.get("model")
        path      = inputs.get("path") or "model.onnx"
        opset     = int(inputs.get("opset") or 17)
        shape_str = inputs.get("input_shape") or "1,784"
        if model is None:
            return {"path": path, "info": "No model connected."}
        try:
            shape = tuple(int(s.strip()) for s in shape_str.split(","))
            dummy = torch.zeros(*shape)
            model.eval()
            torch.onnx.export(
                model, dummy, path,
                opset_version=opset,
                input_names=["input"],
                output_names=["output"],
            )
            info = f"Exported to {path} (opset {opset}, input {shape})"
        except Exception as exc:
            info = f"Export failed: {exc}"
        return {"path": path, "info": info}

    def export(self, iv, ov):
        model  = iv.get("model", "None")
        path   = iv.get("path", repr(self.inputs["path"].default_value))
        opset  = iv.get("opset", "17")
        shape  = iv.get("input_shape", repr(self.inputs["input_shape"].default_value))
        out_path = ov.get("path", "_onnx_path")
        lines = [
            f"_onnx_shape = tuple(int(s.strip()) for s in {shape}.split(','))",
            f"_onnx_dummy = torch.zeros(*_onnx_shape)",
            f"{model}.eval()",
            f"torch.onnx.export({model}, _onnx_dummy, {path},",
            f"    opset_version={opset}, input_names=['input'], output_names=['output'])",
            f"{out_path} = {path}",
        ]
        return ["import torch"], lines
