"""np.ones with shape string node."""
import numpy as np
from core.node import BaseNode, PortType


def _parse_shape(s, default="3,4"):
    s = s or default
    return tuple(int(x) for x in s.split(",") if x.strip())


class NpOnesNode(BaseNode):
    type_name = "np_ones"
    label = "Ones"
    category = "NumPy"
    description = "np.ones with shape string, e.g. '3,4'"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, "3,4")
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            return {"array": np.ones(_parse_shape(inputs.get("shape", "3,4")))}
        except Exception:
            return {"array": None}

    def export(self, iv, ov):
        shape = self._val(iv, "shape")
        return (
            ["import numpy as np"],
            [f"{ov['array']} = np.ones(tuple(int(x) for x in {shape}.split(',') if x.strip()))"],
        )
