"""np.transpose(array) node."""
import numpy as np
from core.node import BaseNode, PortType


class NpTransposeNode(BaseNode):
    type_name = "np_transpose"
    label = "Transpose"
    category = "NumPy"
    description = "np.transpose(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.transpose(arr) if arr is not None else None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['result']} = np.transpose({self._val(iv, 'array')})"],
        )
