"""np.linalg.eig — eigenvalues and eigenvectors node."""
import numpy as np
from core.node import BaseNode, PortType


class NpEigNode(BaseNode):
    type_name = "np_eig"
    label = "Eigenvalues"
    category = "NumPy"
    description = "np.linalg.eig — eigenvalues and eigenvectors"

    def _setup_ports(self):
        self.add_input("matrix",      PortType.NDARRAY)
        self.add_output("eigenvalues",  PortType.NDARRAY)
        self.add_output("eigenvectors", PortType.NDARRAY)

    def execute(self, inputs):
        null = {"eigenvalues": None, "eigenvectors": None}
        try:
            mat = inputs.get("matrix")
            if mat is None:
                return null
            vals, vecs = np.linalg.eig(mat)
            return {"eigenvalues": vals, "eigenvectors": vecs}
        except Exception:
            return null

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['eigenvalues']}, {ov['eigenvectors']} = np.linalg.eig({self._val(iv, 'matrix')})"],
        )
