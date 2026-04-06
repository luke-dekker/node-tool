"""np.linalg.inv(matrix) node."""
import numpy as np
from core.node import BaseNode, PortType


class NpInvNode(BaseNode):
    type_name = "np_inv"
    label = "Inverse"
    category = "NumPy"
    description = "np.linalg.inv(matrix)"

    def _setup_ports(self):
        self.add_input("matrix", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            mat = inputs.get("matrix")
            return {"result": np.linalg.inv(mat) if mat is not None else None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['result']} = np.linalg.inv({self._val(iv, 'matrix')})"],
        )
