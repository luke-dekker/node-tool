"""df.reset_index(drop=True) node."""
from core.node import BaseNode, PortType


class PdResetIndexNode(BaseNode):
    type_name = "pd_reset_index"
    label = "Reset Index"
    category = "Pandas"
    description = "df.reset_index(drop=True)"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"result": df.reset_index(drop=True) if df is not None else None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            [],
            [f"{ov['result']} = {self._val(iv, 'df')}.reset_index(drop=True)"],
        )
