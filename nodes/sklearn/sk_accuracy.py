"""accuracy_score + classification_report node."""
from core.node import BaseNode, PortType


class SkAccuracyNode(BaseNode):
    type_name = "sk_accuracy"
    label = "Accuracy"
    category = "Sklearn"
    description = "accuracy_score + classification_report"

    def _setup_ports(self):
        self.add_input("y_true",    PortType.NDARRAY)
        self.add_input("y_pred",    PortType.NDARRAY)
        self.add_output("accuracy", PortType.FLOAT)
        self.add_output("report",   PortType.STRING)

    def execute(self, inputs):
        null = {"accuracy": None, "report": ""}
        try:
            from sklearn.metrics import accuracy_score, classification_report
            y_true, y_pred = inputs.get("y_true"), inputs.get("y_pred")
            if y_true is None or y_pred is None:
                return null
            return {"accuracy": float(accuracy_score(y_true, y_pred)),
                    "report":   classification_report(y_true, y_pred, zero_division=0)}
        except Exception:
            return null

    def export(self, iv, ov):
        yt = self._val(iv, "y_true")
        yp = self._val(iv, "y_pred")
        return (
            ["from sklearn.metrics import accuracy_score, classification_report"],
            [
                f"{ov['accuracy']} = float(accuracy_score({yt}, {yp}))",
                f"{ov['report']} = classification_report({yt}, {yp}, zero_division=0)",
            ],
        )
