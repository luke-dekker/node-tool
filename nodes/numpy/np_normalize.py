"""(x - min) / (max - min) per array node."""
import numpy as np
from core.node import BaseNode, PortType


class NpNormalizeNode(BaseNode):
    type_name = "np_normalize"
    label = "Normalize"
    category = "NumPy"
    description = "(x - min) / (max - min) per array"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            mn, mx = arr.min(), arr.max()
            rng = mx - mn
            return {"result": (arr - mn) / rng if rng != 0 else np.zeros_like(arr, dtype=float)}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        arr = self._val(iv, "array")
        return (
            ["import numpy as np"],
            [
                f"_mn, _mx = {arr}.min(), {arr}.max()",
                f"{ov['result']} = ({arr} - _mn) / (_mx - _mn) if _mx != _mn else np.zeros_like({arr}, dtype=float)",
            ],
        )
