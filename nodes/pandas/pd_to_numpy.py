"""df.values — convert DataFrame to ndarray node."""
from core.node import BaseNode, PortType


class PdToNumpyNode(BaseNode):
    type_name = "pd_to_numpy"
    label = "To NumPy"
    category = "Pandas"
    description = "df.values — convert DataFrame to ndarray"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"array": df.values if df is not None else None}
        except Exception:
            return {"array": None}

    def export(self, iv, ov):
        return (
            [],
            [f"{ov['array']} = {self._val(iv, 'df')}.values"],
        )
