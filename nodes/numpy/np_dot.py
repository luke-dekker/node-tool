"""np.dot(a, b) node."""
import numpy as np
from core.node import BaseNode, PortType


class NpDotNode(BaseNode):
    type_name = "np_dot"
    label = "Dot"
    category = "NumPy"
    description = "np.dot(a, b)"

    def _setup_ports(self):
        self.add_input("a", PortType.NDARRAY)
        self.add_input("b", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            a, b = inputs.get("a"), inputs.get("b")
            return {"result": np.dot(a, b) if a is not None and b is not None else None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['result']} = np.dot({self._val(iv, 'a')}, {self._val(iv, 'b')})"],
        )
