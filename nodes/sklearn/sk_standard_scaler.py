"""Fit StandardScaler on X_train, transform both splits node."""
from core.node import BaseNode, PortType


class SkStandardScalerNode(BaseNode):
    type_name = "sk_standard_scaler"
    label = "Standard Scaler"
    category = "Sklearn"
    description = "Fit StandardScaler on X_train, transform both splits"

    def _setup_ports(self):
        self.add_input("X_train",         PortType.DATAFRAME)
        self.add_input("X_test",          PortType.DATAFRAME)
        self.add_output("X_train_scaled", PortType.NDARRAY)
        self.add_output("X_test_scaled",  PortType.NDARRAY)
        self.add_output("scaler",         PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"X_train_scaled": None, "X_test_scaled": None, "scaler": None}
        try:
            from sklearn.preprocessing import StandardScaler
            X_train, X_test = inputs.get("X_train"), inputs.get("X_test")
            if X_train is None or X_test is None:
                return null
            sc = StandardScaler()
            return {"X_train_scaled": sc.fit_transform(X_train),
                    "X_test_scaled":  sc.transform(X_test), "scaler": sc}
        except Exception:
            return null

    def export(self, iv, ov):
        Xtr = self._val(iv, "X_train")
        Xte = self._val(iv, "X_test")
        return (
            ["from sklearn.preprocessing import StandardScaler"],
            [
                f"{ov['scaler']} = StandardScaler()",
                f"{ov['X_train_scaled']} = {ov['scaler']}.fit_transform({Xtr})",
                f"{ov['X_test_scaled']} = {ov['scaler']}.transform({Xte})",
            ],
        )
