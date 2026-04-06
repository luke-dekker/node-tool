"""df[col_list] — comma-separated column names node."""
from core.node import BaseNode, PortType


class PdSelectColsNode(BaseNode):
    type_name = "pd_select_cols"
    label = "Select Columns"
    category = "Pandas"
    description = "df[col_list] — comma-separated column names"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_input("columns", PortType.STRING, "col1,col2")
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            cols = [c.strip() for c in (inputs.get("columns") or "").split(",") if c.strip()]
            return {"result": df[cols]}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        df = self._val(iv, "df")
        cols = self._val(iv, "columns")
        return (
            [],
            [
                f"_sel_cols = [c.strip() for c in {cols}.split(',') if c.strip()]",
                f"{ov['result']} = {df}[_sel_cols]",
            ],
        )
