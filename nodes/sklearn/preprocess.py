"""Scikit-learn preprocessing nodes."""
from core.node import BaseNode, PortType

C = "Sklearn"


class SkTrainTestSplitNode(BaseNode):
    type_name = "sk_train_test_split"
    label = "Train/Test Split"
    category = C
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


class SkStandardScalerNode(BaseNode):
    type_name = "sk_standard_scaler"
    label = "Standard Scaler"
    category = C
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


class SkMinMaxScalerNode(BaseNode):
    type_name = "sk_minmax_scaler"
    label = "MinMax Scaler"
    category = C
    description = "Fit MinMaxScaler on X_train, transform both splits"

    def _setup_ports(self):
        self.add_input("X_train",         PortType.DATAFRAME)
        self.add_input("X_test",          PortType.DATAFRAME)
        self.add_output("X_train_scaled", PortType.NDARRAY)
        self.add_output("X_test_scaled",  PortType.NDARRAY)
        self.add_output("scaler",         PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"X_train_scaled": None, "X_test_scaled": None, "scaler": None}
        try:
            from sklearn.preprocessing import MinMaxScaler
            X_train, X_test = inputs.get("X_train"), inputs.get("X_test")
            if X_train is None or X_test is None:
                return null
            sc = MinMaxScaler()
            return {"X_train_scaled": sc.fit_transform(X_train),
                    "X_test_scaled":  sc.transform(X_test), "scaler": sc}
        except Exception:
            return null


class SkLabelEncoderNode(BaseNode):
    type_name = "sk_label_encoder"
    label = "Label Encoder"
    category = C
    description = "Fit LabelEncoder on series, return encoded ndarray + encoder"

    def _setup_ports(self):
        self.add_input("series",   PortType.SERIES)
        self.add_output("encoded", PortType.NDARRAY)
        self.add_output("encoder", PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"encoded": None, "encoder": None}
        try:
            from sklearn.preprocessing import LabelEncoder
            series = inputs.get("series")
            if series is None:
                return null
            enc = LabelEncoder()
            return {"encoded": enc.fit_transform(series), "encoder": enc}
        except Exception:
            return null


class SkOneHotEncoderNode(BaseNode):
    type_name = "sk_onehot_encoder"
    label = "One-Hot Encoder"
    category = C
    description = "OneHotEncoder on ndarray (sparse_output=False)"

    def _setup_ports(self):
        self.add_input("array",    PortType.NDARRAY)
        self.add_output("encoded", PortType.NDARRAY)
        self.add_output("encoder", PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"encoded": None, "encoder": None}
        try:
            import numpy as np
            from sklearn.preprocessing import OneHotEncoder
            array = inputs.get("array")
            if array is None:
                return null
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            enc = OneHotEncoder(sparse_output=False)
            return {"encoded": enc.fit_transform(array), "encoder": enc}
        except Exception:
            return null
