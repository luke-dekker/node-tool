"""np.sum. axis=-99 means over all elements node."""
import numpy as np
from core.node import BaseNode, PortType

_SENTINEL = -99


def _ax(inputs):
    v = inputs.get("axis", _SENTINEL)
    return None if (v is None or int(v) == _SENTINEL) else int(v)


class NpSumNode(BaseNode):
    type_name = "np_sum"
    label = "Sum"
    category = "NumPy"
    description = "np.sum. axis=-99 means over all elements."

    def _setup_ports(self):
        self.add_input("array",  PortType.NDARRAY)
        self.add_input("axis",   PortType.INT, _SENTINEL)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.sum(arr, axis=_ax(inputs)) if arr is not None else None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        ax = self._axis(iv)
        return (
            ["import numpy as np"],
            [f"{ov['result']} = np.sum({self._val(iv, 'array')}, axis={ax})"],
        )
