"""df.groupby(by).agg(agg) node."""
from core.node import BaseNode, PortType


class PdGroupByNode(BaseNode):
    type_name = "pd_groupby"
    label = "Group By"
    category = "Pandas"
    description = "df.groupby(by).agg(agg)"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_input("by",      PortType.STRING, "label")
        self.add_input("agg",     PortType.STRING, "mean")
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            return {"result": df.groupby(inputs.get("by") or "label")
                               .agg(inputs.get("agg") or "mean")
                               .reset_index()}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        df = self._val(iv, "df")
        by = self._val(iv, "by")
        agg = self._val(iv, "agg")
        return (
            [],
            [f"{ov['result']} = {df}.groupby({by}).agg({agg}).reset_index()"],
        )
