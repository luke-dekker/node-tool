"""np.random.randn — shape string, seed>=0 to fix node."""
import numpy as np
from core.node import BaseNode, PortType


def _parse_shape(s, default="3,4"):
    s = s or default
    return tuple(int(x) for x in s.split(",") if x.strip())


class NpRandnNode(BaseNode):
    type_name = "np_randn"
    label = "Random Normal"
    category = "NumPy"
    description = "np.random.randn — shape string, seed>=0 to fix"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, "3,4")
        self.add_input("seed",  PortType.INT,    -1)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            shape = _parse_shape(inputs.get("shape", "3,4"))
            seed  = inputs.get("seed", -1)
            if seed is not None and int(seed) >= 0:
                np.random.seed(int(seed))
            return {"array": np.random.randn(*shape)}
        except Exception:
            return {"array": None}

    def export(self, iv, ov):
        shape = self._val(iv, "shape")
        seed = self._val(iv, "seed")
        return (
            ["import numpy as np"],
            [
                f"_shape = tuple(int(x) for x in {shape}.split(',') if x.strip())",
                f"if int({seed}) >= 0: np.random.seed(int({seed}))",
                f"{ov['array']} = np.random.randn(*_shape)",
            ],
        )
