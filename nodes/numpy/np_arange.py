"""np.arange(start, stop, step) node."""
import numpy as np
from core.node import BaseNode, PortType


class NpArangeNode(BaseNode):
    type_name = "np_arange"
    label = "Arange"
    category = "NumPy"
    description = "np.arange(start, stop, step)"

    def _setup_ports(self):
        self.add_input("start", PortType.FLOAT, 0.0)
        self.add_input("stop",  PortType.FLOAT, 10.0)
        self.add_input("step",  PortType.FLOAT, 1.0)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            start, stop, step = inputs["start"], inputs["stop"], inputs["step"]
            if any(v is None for v in [start, stop, step]):
                return {"array": None}
            return {"array": np.arange(float(start), float(stop), float(step))}
        except Exception:
            return {"array": None}

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['array']} = np.arange({self._val(iv, 'start')}, {self._val(iv, 'stop')}, {self._val(iv, 'step')})"],
        )
