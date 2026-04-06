"""df.corr() — pairwise correlation matrix node."""
from core.node import BaseNode, PortType


class PdCorrelationNode(BaseNode):
    type_name = "pd_correlation"
    label = "Correlation"
    category = "Pandas"
    description = "df.corr() — pairwise correlation matrix"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"result": df.select_dtypes(include="number").corr() if df is not None else None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            [],
            [f"{ov['result']} = {self._val(iv, 'df')}.select_dtypes(include='number').corr()"],
        )
