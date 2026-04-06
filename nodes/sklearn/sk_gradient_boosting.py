"""GradientBoostingClassifier node."""
from core.node import BaseNode, PortType


class SkGradientBoostingNode(BaseNode):
    type_name = "sk_gradient_boosting"
    label = "Gradient Boosting"
    category = "Sklearn"
    description = "GradientBoostingClassifier"

    def _setup_ports(self):
        self.add_input("X_train",       PortType.NDARRAY)
        self.add_input("y_train",       PortType.NDARRAY)
        self.add_input("n_estimators",  PortType.INT,   100)
        self.add_input("learning_rate", PortType.FLOAT, 0.1)
        self.add_output("model",        PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            X, y = inputs.get("X_train"), inputs.get("y_train")
            if X is None or y is None:
                return {"model": None}
            m = GradientBoostingClassifier(n_estimators=int(inputs.get("n_estimators", 100)),
                                           learning_rate=float(inputs.get("learning_rate", 0.1)))
            m.fit(X, y); return {"model": m}
        except Exception:
            return {"model": None}

    def export(self, iv, ov):
        n = self._val(iv, "n_estimators")
        lr = self._val(iv, "learning_rate")
        return (
            ["from sklearn.ensemble import GradientBoostingClassifier"],
            [
                f"{ov['model']} = GradientBoostingClassifier(n_estimators={n}, learning_rate={lr})",
                f"{ov['model']}.fit({self._val(iv, 'X_train')}, {self._val(iv, 'y_train')})",
            ],
        )
