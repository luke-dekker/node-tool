"""Fit LabelEncoder on series, return encoded ndarray + encoder node."""
from core.node import BaseNode, PortType


class SkLabelEncoderNode(BaseNode):
    type_name = "sk_label_encoder"
    label = "Label Encoder"
    category = "Sklearn"
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

    def export(self, iv, ov):
        return (
            ["from sklearn.preprocessing import LabelEncoder"],
            [
                f"{ov['encoder']} = LabelEncoder()",
                f"{ov['encoded']} = {ov['encoder']}.fit_transform({self._val(iv, 'series')})",
            ],
        )
