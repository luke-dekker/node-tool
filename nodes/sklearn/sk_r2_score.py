"""r2_score and mean_squared_error node."""
from core.node import BaseNode, PortType


class SkR2ScoreNode(BaseNode):
    type_name = "sk_r2_score"
    label = "R2 Score"
    category = "Sklearn"
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

    def export(self, iv, ov):
        yt = self._val(iv, "y_true")
        yp = self._val(iv, "y_pred")
        return (
            ["from sklearn.metrics import r2_score, mean_squared_error"],
            [
                f"{ov['r2']} = float(r2_score({yt}, {yp}))",
                f"{ov['mse']} = float(mean_squared_error({yt}, {yp}))",
            ],
        )
