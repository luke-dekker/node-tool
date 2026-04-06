"""model.predict(X) node."""
from core.node import BaseNode, PortType


class SkPredictNode(BaseNode):
    type_name = "sk_predict"
    label = "Predict"
    category = "Sklearn"
    description = "model.predict(X)"

    def _setup_ports(self):
        self.add_input("model",        PortType.SKLEARN_MODEL)
        self.add_input("X",            PortType.NDARRAY)
        self.add_output("predictions", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            model, X = inputs.get("model"), inputs.get("X")
            if model is None or X is None:
                return {"predictions": None}
            return {"predictions": model.predict(X)}
        except Exception:
            return {"predictions": None}

    def export(self, iv, ov):
        return (
            [],
            [f"{ov['predictions']} = {self._val(iv, 'model')}.predict({self._val(iv, 'X')})"],
        )
