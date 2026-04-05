"""NumPy array inspection nodes."""
import numpy as np
from core.node import BaseNode, PortType

C = "NumPy"


class NpArrayInfoNode(BaseNode):
    type_name = "np_array_info"
    label = "Array Info"
    category = C
    description = "Print shape, dtype, min, max, mean of array"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("info", PortType.STRING)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            if arr is None:
                return {"info": "None"}
            a = np.asarray(arr, dtype=float)
            return {"info": (f"shape={list(arr.shape)} dtype={arr.dtype} "
                             f"min={a.min():.4g} max={a.max():.4g} mean={a.mean():.4g}")}
        except Exception:
            return {"info": "error"}


class NpShapeNode(BaseNode):
    type_name = "np_shape"
    label = "Shape"
    category = C
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
