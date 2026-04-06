"""np.eye(n) — identity matrix node."""
import numpy as np
from core.node import BaseNode, PortType


class NpEyeNode(BaseNode):
    type_name = "np_eye"
    label = "Eye"
    category = "NumPy"
    description = "np.eye(n) — identity matrix"

    def _setup_ports(self):
        self.add_input("n", PortType.INT, 3)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            n = inputs.get("n")
            return {"array": np.eye(int(n)) if n is not None else None}
        except Exception:
            return {"array": None}

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['array']} = np.eye({self._val(iv, 'n')})"],
        )
