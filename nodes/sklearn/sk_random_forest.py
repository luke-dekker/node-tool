"""Fit RandomForestClassifier on X_train, y_train node."""
from core.node import BaseNode, PortType


class SkRandomForestNode(BaseNode):
    type_name = "sk_random_forest"
    label = "Random Forest"
    category = "Sklearn"
    description = "Fit RandomForestClassifier on X_train, y_train"

    def _setup_ports(self):
        self.add_input("X_train",      PortType.NDARRAY)
        self.add_input("y_train",      PortType.NDARRAY)
        self.add_input("n_estimators", PortType.INT, 100)
        self.add_input("random_state", PortType.INT, 42)
        self.add_output("model",       PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        try:
            from sklearn.ensemble import RandomForestClassifier
            X, y = inputs.get("X_train"), inputs.get("y_train")
            if X is None or y is None:
                return {"model": None}
            m = RandomForestClassifier(n_estimators=int(inputs.get("n_estimators", 100)),
                                       random_state=int(inputs.get("random_state", 42)))
            m.fit(X, y); return {"model": m}
        except Exception:
            return {"model": None}

    def export(self, iv, ov):
        n = self._val(iv, "n_estimators")
        rs = self._val(iv, "random_state")
        return (
            ["from sklearn.ensemble import RandomForestClassifier"],
            [
                f"{ov['model']} = RandomForestClassifier(n_estimators={n}, random_state={rs})",
                f"{ov['model']}.fit({self._val(iv, 'X_train')}, {self._val(iv, 'y_train')})",
            ],
        )
