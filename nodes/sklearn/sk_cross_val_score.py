"""cross_val_score(model, X, y, cv=cv) node."""
from core.node import BaseNode, PortType


class SkCrossValScoreNode(BaseNode):
    type_name = "sk_cross_val_score"
    label = "Cross-Val Score"
    category = "Sklearn"
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

    def export(self, iv, ov):
        m = self._val(iv, "model")
        X = self._val(iv, "X")
        y = self._val(iv, "y")
        cv = self._val(iv, "cv")
        return (
            ["from sklearn.model_selection import cross_val_score"],
            [
                f"{ov['scores']} = cross_val_score({m}, {X}, {y}, cv={cv})",
                f"{ov['mean']} = float({ov['scores']}.mean())",
            ],
        )
