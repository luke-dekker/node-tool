"""Consolidated sklearn predict node — replaces SkPredictNode + SkPredictProbaNode.

Pick `mode`:
  predict  — model.predict(X) → predictions
  proba    — model.predict_proba(X) → probabilities

Both outputs are NDARRAY; only the relevant one is populated per mode.
"""
from core.node import BaseNode, PortType


_MODES = ["predict", "proba"]


class SkPredictNode(BaseNode):
    type_name   = "sk_predict"
    label       = "Predict"
    category    = "Sklearn"
    description = (
        "Run model.predict(X) or model.predict_proba(X). Pick `mode`."
    )

    def relevant_inputs(self, values):
        return ["mode"]   # model, X are wired

    def _setup_ports(self):
        self.add_input("model",          PortType.SKLEARN_MODEL)
        self.add_input("X",              PortType.NDARRAY)
        self.add_input("mode",           PortType.STRING, "predict", choices=_MODES)
        self.add_output("predictions",   PortType.NDARRAY)
        self.add_output("probabilities", PortType.NDARRAY)

    def execute(self, inputs):
        out = {"predictions": None, "probabilities": None}
        try:
            model, X = inputs.get("model"), inputs.get("X")
            if model is None or X is None:
                return out
            mode = (inputs.get("mode") or "predict").strip()
            if mode == "proba":
                out["probabilities"] = model.predict_proba(X)
            else:
                out["predictions"]   = model.predict(X)
            return out
        except Exception:
            return out

    def export(self, iv, ov):
        m = self._val(iv, "model"); X = self._val(iv, "X")
        mode = (self.inputs["mode"].default_value or "predict")
        if mode == "proba":
            return [], [f"{ov['probabilities']} = {m}.predict_proba({X})"]
        return [], [f"{ov['predictions']} = {m}.predict({X})"]
