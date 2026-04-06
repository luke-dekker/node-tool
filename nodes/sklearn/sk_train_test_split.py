"""sklearn train_test_split on X (DataFrame) and y (Series) node."""
from core.node import BaseNode, PortType


class SkTrainTestSplitNode(BaseNode):
    type_name = "sk_train_test_split"
    label = "Train/Test Split"
    category = "Sklearn"
    description = "sklearn train_test_split on X (DataFrame) and y (Series)"

    def _setup_ports(self):
        self.add_input("X",            PortType.DATAFRAME)
        self.add_input("y",            PortType.SERIES)
        self.add_input("test_size",    PortType.FLOAT, 0.2)
        self.add_input("random_state", PortType.INT,   42)
        self.add_output("X_train",     PortType.DATAFRAME)
        self.add_output("X_test",      PortType.DATAFRAME)
        self.add_output("y_train",     PortType.SERIES)
        self.add_output("y_test",      PortType.SERIES)

    def execute(self, inputs):
        null = {"X_train": None, "X_test": None, "y_train": None, "y_test": None}
        try:
            from sklearn.model_selection import train_test_split
            X, y = inputs.get("X"), inputs.get("y")
            if X is None or y is None:
                return null
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(inputs.get("test_size", 0.2)),
                random_state=int(inputs.get("random_state", 42)))
            return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
        except Exception:
            return null

    def export(self, iv, ov):
        X = self._val(iv, "X")
        y = self._val(iv, "y")
        ts = self._val(iv, "test_size")
        rs = self._val(iv, "random_state")
        return (
            ["from sklearn.model_selection import train_test_split"],
            [
                f"{ov['X_train']}, {ov['X_test']}, {ov['y_train']}, {ov['y_test']} = "
                f"train_test_split({X}, {y}, test_size={ts}, random_state={rs})"
            ],
        )
