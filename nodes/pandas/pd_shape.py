"""Return df.shape as rows, cols node."""
from core.node import BaseNode, PortType


class PdShapeNode(BaseNode):
    type_name = "pd_shape"
    label = "Shape"
    category = "Pandas"
    description = "Return df.shape as rows, cols"

    def _setup_ports(self):
        self.add_input("df",   PortType.DATAFRAME)
        self.add_output("rows", PortType.INT)
        self.add_output("cols", PortType.INT)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"rows": df.shape[0], "cols": df.shape[1]} if df is not None else {"rows": 0, "cols": 0}
        except Exception:
            return {"rows": 0, "cols": 0}

    def export(self, iv, ov):
        df = self._val(iv, "df")
        return (
            [],
            [
                f"{ov['rows']} = {df}.shape[0]",
                f"{ov['cols']} = {df}.shape[1]",
            ],
        )
