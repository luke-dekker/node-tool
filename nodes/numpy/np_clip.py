"""np.clip(array, a_min, a_max) node."""
import numpy as np
from core.node import BaseNode, PortType


class NpClipNode(BaseNode):
    type_name = "np_clip"
    label = "Clip"
    category = "NumPy"
    description = "np.clip(array, a_min, a_max)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("a_min", PortType.FLOAT, 0.0)
        self.add_input("a_max", PortType.FLOAT, 1.0)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            return {"result": np.clip(arr, float(inputs["a_min"]), float(inputs["a_max"]))}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['result']} = np.clip({self._val(iv, 'array')}, {self._val(iv, 'a_min')}, {self._val(iv, 'a_max')})"],
        )
