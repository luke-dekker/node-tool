"""df.head(n).to_string() node."""
from core.node import BaseNode, PortType


class PdHeadNode(BaseNode):
    type_name = "pd_head"
    label = "Head"
    category = "Pandas"
    description = "df.head(n).to_string()"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_input("n",       PortType.INT, 5)
        self.add_output("result", PortType.STRING)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"result": df.head(int(inputs.get("n", 5))).to_string() if df is not None else "None"}
        except Exception:
            return {"result": "error"}

    def export(self, iv, ov):
        return (
            [],
            [f"{ov['result']} = {self._val(iv, 'df')}.head({self._val(iv, 'n')}).to_string()"],
        )
