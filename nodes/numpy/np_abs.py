"""np.abs(array) node."""
import numpy as np
from core.node import BaseNode, PortType


class NpAbsNode(BaseNode):
    type_name = "np_abs"
    label = "Abs"
    category = "NumPy"
    description = "np.abs(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.abs(arr) if arr is not None else None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['result']} = np.abs({self._val(iv, 'array')})"],
        )
