"""Consolidated sklearn preprocessor nodes: SkScalerNode and SkEncoderNode.

Replaces sk_standard_scaler, sk_minmax_scaler, sk_label_encoder, sk_onehot_encoder.
"""
from core.node import BaseNode, PortType


class SkScalerNode(BaseNode):
    type_name = "sk_scaler"
    label = "Scaler"
    category = "Sklearn"
    description = "Fit a scaler on X_train and transform both splits. method: 'standard' or 'minmax'."

    def _setup_ports(self):
        self.add_input("method",          PortType.STRING, "standard")
        self.add_input("X_train",         PortType.DATAFRAME)
        self.add_input("X_test",          PortType.DATAFRAME)
        self.add_output("X_train_scaled", PortType.NDARRAY)
        self.add_output("X_test_scaled",  PortType.NDARRAY)
        self.add_output("scaler",         PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"X_train_scaled": None, "X_test_scaled": None, "scaler": None}
        try:
            X_train = inputs.get("X_train")
            X_test  = inputs.get("X_test")
            if X_train is None or X_test is None:
                return null
            method = str(inputs.get("method") or "standard").strip().lower()
            if method == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                sc = MinMaxScaler()
            else:
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
            return {
                "X_train_scaled": sc.fit_transform(X_train),
                "X_test_scaled":  sc.transform(X_test),
                "scaler":         sc,
            }
        except Exception:
            return null

    def export(self, iv, ov):
        Xtr = self._val(iv, "X_train")
        Xte = self._val(iv, "X_test")
        raw = (self.inputs["method"].default_value
               if iv.get("method") is None else iv["method"])
        if str(raw).strip("'\"").lower() == "minmax":
            cls, imp = "MinMaxScaler", "from sklearn.preprocessing import MinMaxScaler"
        else:
            cls, imp = "StandardScaler", "from sklearn.preprocessing import StandardScaler"
        return (
            [imp],
            [
                f"{ov['scaler']} = {cls}()",
                f"{ov['X_train_scaled']} = {ov['scaler']}.fit_transform({Xtr})",
                f"{ov['X_test_scaled']} = {ov['scaler']}.transform({Xte})",
            ],
        )


class SkEncoderNode(BaseNode):
    type_name = "sk_encoder"
    label = "Encoder"
    category = "Sklearn"
    description = "Encode data with LabelEncoder or OneHotEncoder. method: 'label' or 'onehot'."

    def _setup_ports(self):
        self.add_input("method",   PortType.STRING, "label")
        self.add_input("data",     PortType.NDARRAY)
        self.add_output("encoded", PortType.NDARRAY)
        self.add_output("encoder", PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"encoded": None, "encoder": None}
        try:
            data = inputs.get("data")
            if data is None:
                return null
            if hasattr(data, "values"):
                data = data.values
            method = str(inputs.get("method") or "label").strip().lower()
            if method == "onehot":
                from sklearn.preprocessing import OneHotEncoder
                arr = data.reshape(-1, 1) if data.ndim == 1 else data
                enc = OneHotEncoder(sparse_output=False)
                return {"encoded": enc.fit_transform(arr), "encoder": enc}
            else:
                from sklearn.preprocessing import LabelEncoder
                enc = LabelEncoder()
                return {"encoded": enc.fit_transform(data), "encoder": enc}
        except Exception:
            return null

    def export(self, iv, ov):
        data = self._val(iv, "data")
        raw = (self.inputs["method"].default_value
               if iv.get("method") is None else iv["method"])
        if str(raw).strip("'\"").lower() == "onehot":
            return (
                ["import numpy as np",
                 "from sklearn.preprocessing import OneHotEncoder"],
                [
                    f"_enc_arr = {data}.reshape(-1,1) if {data}.ndim == 1 else {data}",
                    f"{ov['encoder']} = OneHotEncoder(sparse_output=False)",
                    f"{ov['encoded']} = {ov['encoder']}.fit_transform(_enc_arr)",
                ],
            )
        else:
            return (
                ["from sklearn.preprocessing import LabelEncoder"],
                [
                    f"{ov['encoder']} = LabelEncoder()",
                    f"{ov['encoded']} = {ov['encoder']}.fit_transform({data})",
                ],
            )
