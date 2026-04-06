"""Return array shape as string node."""
import numpy as np
from core.node import BaseNode, PortType


class NpShapeNode(BaseNode):
    type_name = "np_shape"
    label = "Shape"
    category = "NumPy"
    description = "Return array shape as string"

    def _setup_ports(self):
        self.add_input("array",  PortType.NDARRAY)
        self.add_output("shape", PortType.STRING)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"shape": str(list(arr.shape)) if arr is not None else "None"}
        except Exception:
            return {"shape": "error"}

    def export(self, iv, ov):
        return (
            [],
            [f"{ov['shape']} = str(list({self._val(iv, 'array')}.shape))"],
        )
