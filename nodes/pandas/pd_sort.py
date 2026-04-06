"""df.sort_values(by, ascending) node."""
from core.node import BaseNode, PortType


class PdSortNode(BaseNode):
    type_name = "pd_sort"
    label = "Sort"
    category = "Pandas"
    description = "df.sort_values(by, ascending)"

    def _setup_ports(self):
        self.add_input("df",        PortType.DATAFRAME)
        self.add_input("by",        PortType.STRING, "col_0")
        self.add_input("ascending", PortType.BOOL,   True)
        self.add_output("result",   PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            return {"result": df.sort_values(by=inputs.get("by") or "col_0",
                                             ascending=bool(inputs.get("ascending", True)))}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        df = self._val(iv, "df")
        by = self._val(iv, "by")
        asc = self._val(iv, "ascending")
        return (
            [],
            [f"{ov['result']} = {df}.sort_values(by={by}, ascending={asc})"],
        )
