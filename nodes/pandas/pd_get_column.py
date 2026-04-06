"""df[column] — returns a Series node."""
from core.node import BaseNode, PortType


class PdGetColumnNode(BaseNode):
    type_name = "pd_get_column"
    label = "Get Column"
    category = "Pandas"
    description = "df[column] — returns a Series"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_input("column", PortType.STRING, "col_0")
        self.add_output("series", PortType.SERIES)

    def execute(self, inputs):
        try:
            df  = inputs.get("df")
            col = inputs.get("column") or "col_0"
            return {"series": df[col] if df is not None else None}
        except Exception:
            return {"series": None}

    def export(self, iv, ov):
        return (
            [],
            [f"{ov['series']} = {self._val(iv, 'df')}[{self._val(iv, 'column')}]"],
        )
