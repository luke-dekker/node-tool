"""df.dropna() node."""
from core.node import BaseNode, PortType


class PdDropNaNode(BaseNode):
    type_name = "pd_dropna"
    label = "Drop NA"
    category = "Pandas"
    description = "df.dropna()"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"result": df.dropna() if df is not None else None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            [],
            [f"{ov['result']} = {self._val(iv, 'df')}.dropna()"],
        )
