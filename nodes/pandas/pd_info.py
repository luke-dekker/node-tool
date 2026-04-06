"""shape, columns, dtypes summary node."""
from core.node import BaseNode, PortType


class PdInfoNode(BaseNode):
    type_name = "pd_info"
    label = "Info"
    category = "Pandas"
    description = "shape, columns, dtypes summary"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_output("result", PortType.STRING)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": "None"}
            dtypes = {col: str(dt) for col, dt in df.dtypes.items()}
            return {"result": f"shape={df.shape} columns={list(df.columns)} dtypes={dtypes}"}
        except Exception:
            return {"result": "error"}

    def export(self, iv, ov):
        df = self._val(iv, "df")
        return (
            [],
            [f"{ov['result']} = f\"shape={{{df}.shape}} columns={{list({df}.columns)}}\""],
        )
