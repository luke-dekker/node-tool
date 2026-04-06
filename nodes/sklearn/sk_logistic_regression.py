"""Fit LogisticRegression on X_train, y_train node."""
from core.node import BaseNode, PortType


class SkLogisticRegressionNode(BaseNode):
    type_name = "sk_logistic_regression"
    label = "Logistic Regression"
    category = "Sklearn"
    description = "Fit LogisticRegression on X_train, y_train"

    def _setup_ports(self):
        self.add_input("X_train",  PortType.NDARRAY)
        self.add_input("y_train",  PortType.NDARRAY)
        self.add_input("max_iter", PortType.INT, 1000)
        self.add_output("model",   PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        try:
            from sklearn.linear_model import LogisticRegression
            X, y = inputs.get("X_train"), inputs.get("y_train")
            if X is None or y is None:
                return {"model": None}
            m = LogisticRegression(max_iter=int(inputs.get("max_iter", 1000)))
            m.fit(X, y); return {"model": m}
        except Exception:
            return {"model": None}

    def export(self, iv, ov):
        return (
            ["from sklearn.linear_model import LogisticRegression"],
            [
                f"{ov['model']} = LogisticRegression(max_iter={self._val(iv, 'max_iter')})",
                f"{ov['model']}.fit({self._val(iv, 'X_train')}, {self._val(iv, 'y_train')})",
            ],
        )
