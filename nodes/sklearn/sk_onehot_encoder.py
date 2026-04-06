"""OneHotEncoder on ndarray (sparse_output=False) node."""
from core.node import BaseNode, PortType


class SkOneHotEncoderNode(BaseNode):
    type_name = "sk_onehot_encoder"
    label = "One-Hot Encoder"
    category = "Sklearn"
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

    def export(self, iv, ov):
        arr = self._val(iv, "array")
        return (
            ["import numpy as np", "from sklearn.preprocessing import OneHotEncoder"],
            [
                f"_ohe_arr = {arr}.reshape(-1,1) if {arr}.ndim == 1 else {arr}",
                f"{ov['encoder']} = OneHotEncoder(sparse_output=False)",
                f"{ov['encoded']} = {ov['encoder']}.fit_transform(_ohe_arr)",
            ],
        )
