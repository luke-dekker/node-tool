"""np.concatenate([a, b], axis=axis) node."""
import numpy as np
from core.node import BaseNode, PortType


class NpConcatNode(BaseNode):
    type_name = "np_concat"
    label = "Concatenate"
    category = "NumPy"
    description = "np.concatenate([a, b], axis=axis)"

    def _setup_ports(self):
        self.add_input("a",    PortType.NDARRAY)
        self.add_input("b",    PortType.NDARRAY)
        self.add_input("axis", PortType.INT, 0)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            a, b = inputs.get("a"), inputs.get("b")
            if a is None or b is None:
                return {"result": None}
            return {"result": np.concatenate([a, b], axis=int(inputs.get("axis", 0)))}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['result']} = np.concatenate([{self._val(iv, 'a')}, {self._val(iv, 'b')}], axis={self._val(iv, 'axis')})"],
        )
