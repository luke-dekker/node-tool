"""pd.merge(left, right, on=on, how=how) node."""
from core.node import BaseNode, PortType


class PdMergeNode(BaseNode):
    type_name = "pd_merge"
    label = "Merge"
    category = "Pandas"
    description = "pd.merge(left, right, on=on, how=how)"

    def _setup_ports(self):
        self.add_input("left",    PortType.DATAFRAME)
        self.add_input("right",   PortType.DATAFRAME)
        self.add_input("on",      PortType.STRING,   "index")
        self.add_input("how",     PortType.STRING,   "inner")
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import pandas as pd
            left  = inputs.get("left")
            right = inputs.get("right")
            on    = inputs.get("on")  or "index"
            how   = inputs.get("how") or "inner"
            if left is None or right is None:
                return {"result": None}
            if on == "index":
                return {"result": pd.merge(left, right, left_index=True, right_index=True, how=how)}
            return {"result": pd.merge(left, right, on=on, how=how)}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        l = self._val(iv, "left")
        r = self._val(iv, "right")
        on = self._val(iv, "on")
        how = self._val(iv, "how")
        return (
            ["import pandas as pd"],
            [f"{ov['result']} = pd.merge({l}, {r}, on={on}, how={how})"],
        )
