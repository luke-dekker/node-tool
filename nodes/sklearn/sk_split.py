"""Consolidated train/test split + cross-val node.

Replaces SkTrainTestSplitNode + SkCrossValScoreNode. Pick `mode`:
  train_test  — outputs X_train, X_test, y_train, y_test
  cross_val   — outputs scores, mean_score (uses `model` input)
"""
from core.node import BaseNode, PortType


_MODES = ["train_test", "cross_val"]


class SkSplitNode(BaseNode):
    type_name   = "sk_split"
    label       = "Train Split / CV"
    category    = "Sklearn"
    description = (
        "Pick `mode`:\n"
        "  train_test — sklearn train_test_split → X_train, X_test, y_train, y_test\n"
        "  cross_val  — cross_val_score(model, X, y, cv) → scores, mean_score"
    )

    def relevant_inputs(self, values):
        mode = (values.get("mode") or "train_test").strip()
        if mode == "train_test": return ["mode", "test_size", "random_state"]
        if mode == "cross_val":  return ["mode", "cv"]
        return ["mode"]

    def _setup_ports(self):
        self.add_input("X",            PortType.DATAFRAME)
        self.add_input("y",            PortType.SERIES)
        self.add_input("mode",         PortType.STRING, "train_test", choices=_MODES)
        self.add_input("model",        PortType.SKLEARN_MODEL, optional=True)
        self.add_input("test_size",    PortType.FLOAT, 0.2, optional=True)
        self.add_input("random_state", PortType.INT,   42,  optional=True)
        self.add_input("cv",           PortType.INT,   5,   optional=True)
        # train_test outputs
        self.add_output("X_train",    PortType.DATAFRAME)
        self.add_output("X_test",     PortType.DATAFRAME)
        self.add_output("y_train",    PortType.SERIES)
        self.add_output("y_test",     PortType.SERIES)
        # cross_val outputs
        self.add_output("scores",     PortType.NDARRAY)
        self.add_output("mean_score", PortType.FLOAT)

    def execute(self, inputs):
        out = {"X_train": None, "X_test": None, "y_train": None, "y_test": None,
               "scores": None, "mean_score": 0.0}
        try:
            X, y = inputs.get("X"), inputs.get("y")
            if X is None or y is None:
                return out
            mode = (inputs.get("mode") or "train_test").strip()
            if mode == "train_test":
                from sklearn.model_selection import train_test_split
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=float(inputs.get("test_size", 0.2)),
                    random_state=int(inputs.get("random_state", 42)))
                out.update({"X_train": X_tr, "X_test": X_te,
                            "y_train": y_tr, "y_test": y_te})
                return out
            if mode == "cross_val":
                from sklearn.model_selection import cross_val_score
                model = inputs.get("model")
                if model is None:
                    return out
                scores = cross_val_score(model, X, y, cv=int(inputs.get("cv", 5)))
                out["scores"] = scores
                out["mean_score"] = float(scores.mean())
                return out
            return out
        except Exception:
            return out

    def export(self, iv, ov):
        X = self._val(iv, "X"); y = self._val(iv, "y")
        mode = (self.inputs["mode"].default_value or "train_test")
        if mode == "train_test":
            ts = self._val(iv, "test_size"); rs = self._val(iv, "random_state")
            return ["from sklearn.model_selection import train_test_split"], [
                f"{ov['X_train']}, {ov['X_test']}, {ov['y_train']}, {ov['y_test']} = "
                f"train_test_split({X}, {y}, test_size={ts}, random_state={rs})"
            ]
        if mode == "cross_val":
            m = self._val(iv, "model"); cv = self._val(iv, "cv")
            return ["from sklearn.model_selection import cross_val_score"], [
                f"{ov['scores']} = cross_val_score({m}, {X}, {y}, cv={cv})",
                f"{ov['mean_score']} = float({ov['scores']}.mean())",
            ]
        return [], [f"# unknown split mode {mode!r}"]
