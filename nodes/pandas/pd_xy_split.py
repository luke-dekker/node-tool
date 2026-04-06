"""Split DataFrame into features X and label series y node."""
from core.node import BaseNode, PortType


class PdXYSplitNode(BaseNode):
    type_name = "pd_xy_split"
    label = "XY Split"
    category = "Pandas"
    description = "Split DataFrame into features X and label series y"

    def _setup_ports(self):
        self.add_input("df",        PortType.DATAFRAME)
        self.add_input("label_col", PortType.STRING,   "label")
        self.add_output("X",        PortType.DATAFRAME)
        self.add_output("y",        PortType.SERIES)

    def execute(self, inputs):
        null = {"X": None, "y": None}
        try:
            df  = inputs.get("df")
            col = inputs.get("label_col") or "label"
            if df is None:
                return null
            return {"X": df.drop(columns=[col]), "y": df[col]}
        except Exception:
            return null

    def export(self, iv, ov):
        df = self._val(iv, "df")
        lbl = self._val(iv, "label_col")
        return (
            [],
            [
                f"{ov['y']} = {df}[{lbl}]",
                f"{ov['X']} = {df}.drop(columns=[{lbl}])",
            ],
        )
