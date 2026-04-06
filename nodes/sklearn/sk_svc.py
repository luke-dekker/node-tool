"""Fit SVC on X_train, y_train node."""
from core.node import BaseNode, PortType


class SkSVCNode(BaseNode):
    type_name = "sk_svc"
    label = "SVC"
    category = "Sklearn"
    description = "Fit SVC on X_train, y_train"

    def _setup_ports(self):
        self.add_input("X_train", PortType.NDARRAY)
        self.add_input("y_train", PortType.NDARRAY)
        self.add_input("C",       PortType.FLOAT,  1.0)
        self.add_input("kernel",  PortType.STRING, "rbf")
        self.add_output("model",  PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        try:
            from sklearn.svm import SVC
            X, y = inputs.get("X_train"), inputs.get("y_train")
            if X is None or y is None:
                return {"model": None}
            m = SVC(C=float(inputs.get("C", 1.0)), kernel=inputs.get("kernel") or "rbf")
            m.fit(X, y); return {"model": m}
        except Exception:
            return {"model": None}

    def export(self, iv, ov):
        C = self._val(iv, "C")
        k = self._val(iv, "kernel")
        return (
            ["from sklearn.svm import SVC"],
            [
                f"{ov['model']} = SVC(C={C}, kernel={k})",
                f"{ov['model']}.fit({self._val(iv, 'X_train')}, {self._val(iv, 'y_train')})",
            ],
        )
