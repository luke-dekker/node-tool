"""Fit LinearRegression on X_train, y_train node."""
from core.node import BaseNode, PortType


class SkLinearRegressionNode(BaseNode):
    type_name = "sk_linear_regression"
    label = "Linear Regression"
    category = "Sklearn"
    description = "Fit LinearRegression on X_train, y_train"

    def _setup_ports(self):
        self.add_input("X_train", PortType.NDARRAY)
        self.add_input("y_train", PortType.NDARRAY)
        self.add_output("model",  PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        try:
            from sklearn.linear_model import LinearRegression
            X, y = inputs.get("X_train"), inputs.get("y_train")
            if X is None or y is None:
                return {"model": None}
            m = LinearRegression(); m.fit(X, y)
            return {"model": m}
        except Exception:
            return {"model": None}

    def export(self, iv, ov):
        return (
            ["from sklearn.linear_model import LinearRegression"],
            [
                f"{ov['model']} = LinearRegression()",
                f"{ov['model']}.fit({self._val(iv, 'X_train')}, {self._val(iv, 'y_train')})",
            ],
        )
