"""np.matmul(a, b) node."""
import numpy as np
from core.node import BaseNode, PortType


class NpMatMulNode(BaseNode):
    type_name = "np_matmul"
    label = "MatMul"
    category = "NumPy"
    description = "np.matmul(a, b)"

    def _setup_ports(self):
        self.add_input("a", PortType.NDARRAY)
        self.add_input("b", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            a, b = inputs.get("a"), inputs.get("b")
            return {"result": np.matmul(a, b) if a is not None and b is not None else None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['result']} = np.matmul({self._val(iv, 'a')}, {self._val(iv, 'b')})"],
        )
