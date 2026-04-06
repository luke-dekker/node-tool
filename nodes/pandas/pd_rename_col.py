"""Rename a single column old_name -> new_name node."""
from core.node import BaseNode, PortType


class PdRenameColNode(BaseNode):
    type_name = "pd_rename_col"
    label = "Rename Column"
    category = "Pandas"
    description = "Rename a single column old_name -> new_name"

    def _setup_ports(self):
        self.add_input("df",       PortType.DATAFRAME)
        self.add_input("old_name", PortType.STRING, "col_0")
        self.add_input("new_name", PortType.STRING, "x")
        self.add_output("result",  PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            return {"result": df.rename(columns={inputs.get("old_name") or "col_0":
                                                  inputs.get("new_name") or "x"})}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        df = self._val(iv, "df")
        old = self._val(iv, "old_name")
        new = self._val(iv, "new_name")
        return (
            [],
            [f"{ov['result']} = {df}.rename(columns={{{old}: {new}}})"],
        )
