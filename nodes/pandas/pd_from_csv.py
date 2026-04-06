"""pd.read_csv(path, sep=sep) node."""
from core.node import BaseNode, PortType


class PdFromCsvNode(BaseNode):
    type_name = "pd_from_csv"
    label = "From CSV"
    category = "Pandas"
    description = "pd.read_csv(path, sep=sep)"

    def _setup_ports(self):
        self.add_input("path", PortType.STRING, "data.csv")
        self.add_input("sep",  PortType.STRING, ",")
        self.add_output("df",  PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import pandas as pd
            return {"df": pd.read_csv(inputs.get("path") or "data.csv",
                                      sep=inputs.get("sep") or ",")}
        except Exception:
            return {"df": None}

    def export(self, iv, ov):
        return (
            ["import pandas as pd"],
            [f"{ov['df']} = pd.read_csv({self._val(iv, 'path')}, sep={self._val(iv, 'sep')})"],
        )
