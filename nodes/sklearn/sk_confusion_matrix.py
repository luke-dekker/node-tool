"""confusion_matrix(y_true, y_pred) node."""
from core.node import BaseNode, PortType


class SkConfusionMatrixNode(BaseNode):
    type_name = "sk_confusion_matrix"
    label = "Confusion Matrix"
    category = "Sklearn"
    description = "confusion_matrix(y_true, y_pred)"

    def _setup_ports(self):
        self.add_input("y_true",  PortType.NDARRAY)
        self.add_input("y_pred",  PortType.NDARRAY)
        self.add_output("matrix", PortType.NDARRAY)
        self.add_output("info",   PortType.STRING)

    def execute(self, inputs):
        null = {"matrix": None, "info": ""}
        try:
            from sklearn.metrics import confusion_matrix
            y_true, y_pred = inputs.get("y_true"), inputs.get("y_pred")
            if y_true is None or y_pred is None:
                return null
            mat = confusion_matrix(y_true, y_pred)
            return {"matrix": mat, "info": f"Confusion matrix shape: {mat.shape}"}
        except Exception:
            return null

    def export(self, iv, ov):
        yt = self._val(iv, "y_true")
        yp = self._val(iv, "y_pred")
        return (
            ["from sklearn.metrics import confusion_matrix"],
            [
                f"{ov['matrix']} = confusion_matrix({yt}, {yp})",
                f"{ov['info']} = 'Confusion matrix shape: ' + str({ov['matrix']}.shape)",
            ],
        )
