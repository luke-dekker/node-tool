"""df.describe().to_string() node."""
from core.node import BaseNode, PortType


class PdDescribeNode(BaseNode):
    type_name = "pd_describe"
    label = "Describe"
    category = "Pandas"
    description = "df.describe().to_string()"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_output("result", PortType.STRING)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"result": df.describe().to_string() if df is not None else "None"}
        except Exception:
            return {"result": "error"}

    def export(self, iv, ov):
        return (
            [],
            [f"{ov['result']} = {self._val(iv, 'df')}.describe().to_string()"],
        )
