"""np.linspace(start, stop, num) node."""
import numpy as np
from core.node import BaseNode, PortType


class NpLinspaceNode(BaseNode):
    type_name = "np_linspace"
    label = "Linspace"
    category = "NumPy"
    description = "np.linspace(start, stop, num)"

    def _setup_ports(self):
        self.add_input("start", PortType.FLOAT, 0.0)
        self.add_input("stop",  PortType.FLOAT, 1.0)
        self.add_input("num",   PortType.INT,   50)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            start, stop, num = inputs["start"], inputs["stop"], inputs["num"]
            if any(v is None for v in [start, stop, num]):
                return {"array": None}
            return {"array": np.linspace(float(start), float(stop), int(num))}
        except Exception:
            return {"array": None}

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['array']} = np.linspace({self._val(iv, 'start')}, {self._val(iv, 'stop')}, {self._val(iv, 'num')})"],
        )
