"""Consolidated sklearn metrics node — replaces SkAccuracyNode, SkR2ScoreNode,
SkConfusionMatrixNode.

Pick `metric`:
  accuracy          — classification accuracy + classification_report
  r2                — regression r2 + mean_squared_error
  confusion_matrix  — confusion matrix array

Outputs (all 4 always present; unused ones are None / 0 / ""):
  value     — primary float    (accuracy → score, r2 → r2, confusion_matrix → 0.0)
  secondary — secondary float  (r2 → mse, accuracy → 0.0, confusion_matrix → 0.0)
  matrix    — np.ndarray       (confusion_matrix only)
  info      — str              (accuracy → classification_report, others → summary)
"""
from __future__ import annotations
from core.node import BaseNode, PortType


_METRICS = ["accuracy", "r2", "confusion_matrix"]


class SkMetricsNode(BaseNode):
    type_name   = "sk_metrics"
    label       = "Metrics"
    category    = "Sklearn"
    description = (
        "Compute sklearn metrics. Pick `metric`:\n"
        "  accuracy          — value=accuracy, info=classification_report\n"
        "  r2                — value=r2, secondary=mse\n"
        "  confusion_matrix  — matrix=confusion array, info=shape summary"
    )

    def _setup_ports(self):
        self.add_input("y_true",   PortType.NDARRAY)
        self.add_input("y_pred",   PortType.NDARRAY)
        self.add_input("metric",   PortType.STRING, "accuracy", choices=_METRICS)
        self.add_output("value",     PortType.FLOAT)
        self.add_output("secondary", PortType.FLOAT)
        self.add_output("matrix",    PortType.NDARRAY)
        self.add_output("info",      PortType.STRING)

    def relevant_inputs(self, values):
        return ["metric"]   # y_true, y_pred are wired data ports

    def execute(self, inputs):
        out = {"value": 0.0, "secondary": 0.0, "matrix": None, "info": ""}
        try:
            y_true, y_pred = inputs.get("y_true"), inputs.get("y_pred")
            if y_true is None or y_pred is None:
                return out
            metric = (inputs.get("metric") or "accuracy").strip()
            if metric == "accuracy":
                from sklearn.metrics import accuracy_score, classification_report
                out["value"] = float(accuracy_score(y_true, y_pred))
                out["info"]  = classification_report(y_true, y_pred, zero_division=0)
                return out
            if metric == "r2":
                from sklearn.metrics import r2_score, mean_squared_error
                out["value"]     = float(r2_score(y_true, y_pred))
                out["secondary"] = float(mean_squared_error(y_true, y_pred))
                return out
            if metric == "confusion_matrix":
                from sklearn.metrics import confusion_matrix
                mat = confusion_matrix(y_true, y_pred)
                out["matrix"] = mat
                out["info"]   = f"Confusion matrix shape: {mat.shape}"
                return out
            return out
        except Exception:
            return out

    def export(self, iv, ov):
        yt = self._val(iv, "y_true")
        yp = self._val(iv, "y_pred")
        metric = (self.inputs["metric"].default_value or "accuracy")
        if metric == "accuracy":
            return ["from sklearn.metrics import accuracy_score, classification_report"], [
                f"{ov['value']} = float(accuracy_score({yt}, {yp}))",
                f"{ov['info']}  = classification_report({yt}, {yp}, zero_division=0)",
            ]
        if metric == "r2":
            return ["from sklearn.metrics import r2_score, mean_squared_error"], [
                f"{ov['value']}     = float(r2_score({yt}, {yp}))",
                f"{ov['secondary']} = float(mean_squared_error({yt}, {yp}))",
            ]
        if metric == "confusion_matrix":
            return ["from sklearn.metrics import confusion_matrix"], [
                f"{ov['matrix']} = confusion_matrix({yt}, {yp})",
                f"{ov['info']}   = 'Confusion matrix shape: ' + str({ov['matrix']}.shape)",
            ]
        return [], [f"# unknown metric {metric!r}"]
