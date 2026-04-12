"""Print Tensor node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class PrintTensorNode(BaseNode):
    type_name   = "pt_print_tensor"
    label       = "Print Tensor"
    category    = "Tensor Ops"
    subcategory = ""
    description = "Log tensor shape to terminal and pass through"

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("label", PortType.STRING, default="")
        self.add_output("passthrough", PortType.TENSOR)
        self.add_output("__terminal__", PortType.STRING)

    def execute(self, inputs):
        try:
            tensor = inputs.get("tensor")
            label = inputs.get("label", "") or ""
            if tensor is None:
                return {"passthrough": None, "__terminal__": f"{label}: None"}
            shape = list(tensor.shape)
            n = tensor.numel()
            if n <= 128:
                vals = tensor.flatten().tolist()
                vals_str = ", ".join(
                    str(int(v)) if float(v) == int(v) else f"{v:.4g}" for v in vals
                )
                msg = f"{label}: shape={shape}  [{vals_str}]"
            else:
                msg = f"{label}: shape={shape} dtype={tensor.dtype}"
            return {"passthrough": tensor, "__terminal__": msg}
        except Exception:
            return {"passthrough": None, "__terminal__": "error"}

    def export(self, iv, ov):
        t = self._val(iv, 'tensor')
        lbl = self._val(iv, 'label')
        return [], [
            f"{ov['passthrough']} = {t}",
            f"print(f\"{{{lbl}}}: shape={{list({t}.shape)}}\" if {lbl} else f\"shape={{list({t}.shape)}}\")",
        ]
