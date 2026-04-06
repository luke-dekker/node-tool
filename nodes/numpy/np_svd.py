"""np.linalg.svd — U, S, Vt node."""
import numpy as np
from core.node import BaseNode, PortType


class NpSVDNode(BaseNode):
    type_name = "np_svd"
    label = "SVD"
    category = "NumPy"
    description = "np.linalg.svd — U, S, Vt"

    def _setup_ports(self):
        self.add_input("matrix", PortType.NDARRAY)
        self.add_output("U",  PortType.NDARRAY)
        self.add_output("S",  PortType.NDARRAY)
        self.add_output("Vt", PortType.NDARRAY)

    def execute(self, inputs):
        null = {"U": None, "S": None, "Vt": None}
        try:
            mat = inputs.get("matrix")
            if mat is None:
                return null
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)
            return {"U": U, "S": S, "Vt": Vt}
        except Exception:
            return null

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['U']}, {ov['S']}, {ov['Vt']} = np.linalg.svd({self._val(iv, 'matrix')}, full_matrices=False)"],
        )
