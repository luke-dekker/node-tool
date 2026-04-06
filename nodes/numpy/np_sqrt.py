"""np.sqrt(array) node."""
import numpy as np
from core.node import BaseNode, PortType


class NpSqrtNode(BaseNode):
    type_name = "np_sqrt"
    label = "Sqrt"
    category = "NumPy"
    description = "np.sqrt(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.sqrt(arr) if arr is not None else None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['result']} = np.sqrt({self._val(iv, 'array')})"],
        )
