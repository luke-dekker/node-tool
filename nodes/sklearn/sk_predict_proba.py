"""model.predict_proba(X) node."""
from core.node import BaseNode, PortType


class SkPredictProbaNode(BaseNode):
    type_name = "sk_predict_proba"
    label = "Predict Proba"
    category = "Sklearn"
    description = "model.predict_proba(X)"

    def _setup_ports(self):
        self.add_input("model",          PortType.SKLEARN_MODEL)
        self.add_input("X",              PortType.NDARRAY)
        self.add_output("probabilities", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            model, X = inputs.get("model"), inputs.get("X")
            if model is None or X is None:
                return {"probabilities": None}
            return {"probabilities": model.predict_proba(X)}
        except Exception:
            return {"probabilities": None}

    def export(self, iv, ov):
        return (
            [],
            [f"{ov['probabilities']} = {self._val(iv, 'model')}.predict_proba({self._val(iv, 'X')})"],
        )
