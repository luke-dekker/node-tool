"""np.log(array) — clipped to avoid -inf node."""
import numpy as np
from core.node import BaseNode, PortType


class NpLogNode(BaseNode):
    type_name = "np_log"
    label = "Log"
    category = "NumPy"
    description = "np.log(array) — clipped to avoid -inf"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.log(np.clip(arr, 1e-12, None)) if arr is not None else None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        arr = self._val(iv, "array")
        return (
            ["import numpy as np"],
            [f"{ov['result']} = np.log(np.clip({arr}, 1e-12, None))"],
        )
