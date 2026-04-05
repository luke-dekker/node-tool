"""Scikit-learn metrics nodes."""
from core.node import BaseNode, PortType

C = "Sklearn"


class SkAccuracyNode(BaseNode):
    type_name = "sk_accuracy"
    label = "Accuracy"
    category = C
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


class SkConfusionMatrixNode(BaseNode):
    type_name = "sk_confusion_matrix"
    label = "Confusion Matrix"
    category = C
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


class SkR2ScoreNode(BaseNode):
    type_name = "sk_r2_score"
    label = "R2 Score"
    category = C
    description = "r2_score and mean_squared_error"

    def _setup_ports(self):
        self.add_input("y_true", PortType.NDARRAY)
        self.add_input("y_pred", PortType.NDARRAY)
        self.add_output("r2",    PortType.FLOAT)
        self.add_output("mse",   PortType.FLOAT)

    def execute(self, inputs):
        null = {"r2": None, "mse": None}
        try:
            from sklearn.metrics import r2_score, mean_squared_error
            y_true, y_pred = inputs.get("y_true"), inputs.get("y_pred")
            if y_true is None or y_pred is None:
                return null
            return {"r2":  float(r2_score(y_true, y_pred)),
                    "mse": float(mean_squared_error(y_true, y_pred))}
        except Exception:
            return null


class SkCrossValScoreNode(BaseNode):
    type_name = "sk_cross_val_score"
    label = "Cross-Val Score"
    category = C
    description = "cross_val_score(model, X, y, cv=cv)"

    def _setup_ports(self):
        self.add_input("model",   PortType.SKLEARN_MODEL)
        self.add_input("X",       PortType.NDARRAY)
        self.add_input("y",       PortType.NDARRAY)
        self.add_input("cv",      PortType.INT, 5)
        self.add_output("scores", PortType.NDARRAY)
        self.add_output("mean",   PortType.FLOAT)

    def execute(self, inputs):
        null = {"scores": None, "mean": None}
        try:
            from sklearn.model_selection import cross_val_score
            model, X, y = inputs.get("model"), inputs.get("X"), inputs.get("y")
            if model is None or X is None or y is None:
                return null
            scores = cross_val_score(model, X, y, cv=int(inputs.get("cv", 5)))
            return {"scores": scores, "mean": float(scores.mean())}
        except Exception:
            return null
