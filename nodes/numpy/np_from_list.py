"""Parse comma-separated values string into ndarray node."""
import numpy as np
from core.node import BaseNode, PortType


class NpFromListNode(BaseNode):
    type_name = "np_from_list"
    label = "From List"
    category = "NumPy"
    description = "Parse comma-separated values string into ndarray."

    def _setup_ports(self):
        self.add_input("values", PortType.STRING, "1,2,3,4")
        self.add_input("dtype",  PortType.STRING, "float32")
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            vals  = [float(x) for x in (inputs.get("values") or "").split(",") if x.strip()]
            dtype = inputs.get("dtype") or "float32"
            return {"array": np.array(vals, dtype=dtype)}
        except Exception:
            return {"array": None}

    def export(self, iv, ov):
        vals = self._val(iv, "values")
        dtype = self._val(iv, "dtype")
        return (
            ["import numpy as np"],
            [f"{ov['array']} = np.array([float(x) for x in {vals}.split(',') if x.strip()], dtype={dtype})"],
        )
