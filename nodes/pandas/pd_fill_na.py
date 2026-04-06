"""df.fillna(value) node."""
from core.node import BaseNode, PortType


class PdFillNaNode(BaseNode):
    type_name = "pd_fillna"
    label = "Fill NA"
    category = "Pandas"
    description = "df.fillna(value)"

    def _setup_ports(self):
        self.add_input("df",    PortType.DATAFRAME)
        self.add_input("value", PortType.FLOAT, 0.0)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"result": df.fillna(float(inputs.get("value", 0.0))) if df is not None else None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            [],
            [f"{ov['result']} = {self._val(iv, 'df')}.fillna({self._val(iv, 'value')})"],
        )
