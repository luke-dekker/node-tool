"""pd.DataFrame(array) — optional comma-separated column names node."""
from core.node import BaseNode, PortType


class PdFromNumpyNode(BaseNode):
    type_name = "pd_from_numpy"
    label = "From NumPy"
    category = "Pandas"
    description = "pd.DataFrame(array) — optional comma-separated column names"

    def _setup_ports(self):
        self.add_input("array",   PortType.NDARRAY)
        self.add_input("columns", PortType.STRING, "")
        self.add_output("df",     PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import pandas as pd
            array   = inputs.get("array")
            col_str = inputs.get("columns") or ""
            if array is None:
                return {"df": None}
            cols = [c.strip() for c in col_str.split(",") if c.strip()] or None
            return {"df": pd.DataFrame(array, columns=cols)}
        except Exception:
            return {"df": None}

    def export(self, iv, ov):
        arr = self._val(iv, "array")
        cols = self._val(iv, "columns")
        return (
            ["import pandas as pd"],
            [
                f"_cols = [{cols}] if {cols}.strip() else None",
                f"{ov['df']} = pd.DataFrame({arr}, columns=_cols)",
            ],
        )
