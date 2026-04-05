"""NumPy linear algebra nodes."""
import numpy as np
from core.node import BaseNode, PortType

C = "NumPy"


class NpDotNode(BaseNode):
    type_name = "np_dot"
    label = "Dot"
    category = C
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


class NpMatMulNode(BaseNode):
    type_name = "np_matmul"
    label = "MatMul"
    category = C
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


class NpInvNode(BaseNode):
    type_name = "np_inv"
    label = "Inverse"
    category = C
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


class NpEigNode(BaseNode):
    type_name = "np_eig"
    label = "Eigenvalues"
    category = C
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


class NpSVDNode(BaseNode):
    type_name = "np_svd"
    label = "SVD"
    category = C
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
