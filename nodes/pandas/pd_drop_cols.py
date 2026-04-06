"""df.drop(columns=[...]) — comma-separated node."""
from core.node import BaseNode, PortType


class PdDropColsNode(BaseNode):
    type_name = "pd_drop_cols"
    label = "Drop Columns"
    category = "Pandas"
    description = "df.drop(columns=[...]) — comma-separated"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_input("columns", PortType.STRING, "col1")
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            cols = [c.strip() for c in (inputs.get("columns") or "").split(",") if c.strip()]
            return {"result": df.drop(columns=cols)}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        df = self._val(iv, "df")
        cols = self._val(iv, "columns")
        return (
            [],
            [
                f"_drop_cols = [c.strip() for c in {cols}.split(',') if c.strip()]",
                f"{ov['result']} = {df}.drop(columns=_drop_cols)",
            ],
        )
